// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_PERF_COUNTERS_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_PERF_COUNTERS_H_

#include "iree/base/api.h"
#include "iree/hal/driver/amd/gpu/target/identity.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Hardware performance counters (perfmon) for one PM4 compute publication.
// The start sequence programs the shader-queue counters and starts them, a
// sample sequence latches every counter and copies it to host-visible memory,
// and the stop sequence stops counting and returns every register it wrote to
// its reset default. A dispatch's value is the difference between the samples
// taken before and after it.
//
// The perfmon registers are device-wide, not queue context: one collector at a
// time, and a second one (rocprofv3 --pmc, RADV performance queries)
// reprograms the same registers. What is counted is narrower. The hardware
// counts a queue only while that queue's own COMPUTE_PERFCOUNT_ENABLE is set,
// within the per-VMID mask of each block's CTRL2 register. The sequences set
// the enable in start and clear it in stop, and never set FORCE_EN, which
// would count every queue on the device. Every queue of the process that has
// the enable set is counted together.
//
// Two kinds of counter exist. Engine counters (SQG) have one 64-bit instance
// per shader engine. Instance counters (SQ) have one 32-bit instance per
// workgroup processor. Both are read per instance; the host sums them.

enum {
  // Shader engines addressable by one counter configuration.
  IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_MAX_ENGINE_COUNT = 8,
  // Engine counters selectable at once: the hardware has eight SQG slots.
  IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_MAX_ENGINE_COUNTERS = 8,
  // Instance counters selectable at once. Only even SQ select registers are
  // usable and select 2k feeds result register k; the first four are
  // qualified.
  IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_MAX_INSTANCE_COUNTERS = 4,
  // Workgroup processors addressable by one counter configuration.
  IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_MAX_INSTANCE_COUNT = 64,
  // DWORDs the start sequence copies before it writes anything: the perfmon
  // control, the engine block control and the first engine select.
  IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_STATE_WORD_COUNT = 4,
  // Host-written pattern of a raw sample DWORD no copy has replaced.
  IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_UNWRITTEN = (int)0xffffffff,
};

// Hardware block owning a counter.
typedef enum iree_hal_amd_gpu_pm4_perf_counter_block_e {
  // SQG: one 64-bit instance per shader engine.
  IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_BLOCK_ENGINE = 0,
  // SQ: one 32-bit instance per workgroup processor.
  IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_BLOCK_INSTANCE = 1,
} iree_hal_amd_gpu_pm4_perf_counter_block_t;

// One selectable hardware event.
typedef struct iree_hal_amd_gpu_pm4_perf_counter_event_t {
  // Counter name as requested and emitted, e.g. SQG_WAVES.
  iree_string_view_t name;
  // Hardware block name: SQG or SQ.
  iree_string_view_t block_name;
  // What the event counts.
  iree_string_view_t description;
  // Block owning the event.
  iree_hal_amd_gpu_pm4_perf_counter_block_t block;
  // Event number written to the block's select register.
  uint32_t select;
  // Values are clock cycles instead of event counts.
  bool cycles;
} iree_hal_amd_gpu_pm4_perf_counter_event_t;

// Target register table, control values and event catalog. Register fields
// are byte addresses, except compute_enable which is an SH register index.
typedef struct iree_hal_amd_gpu_pm4_perf_counter_profile_t {
  // GRBM_GFX_INDEX and its full-broadcast value.
  uint32_t graphics_index;
  uint32_t graphics_index_broadcast;
  // CP_PERFMON_CNTL and the values that clear, start, sample and stop.
  uint32_t control;
  uint32_t control_clear;
  uint32_t control_start;
  uint32_t control_sample;
  uint32_t control_stop;
  // SQ_PERFCOUNTER_CTRL and SQG_PERFCOUNTER_CTRL with their stage enables.
  uint32_t instance_block_control;
  uint32_t engine_block_control;
  uint32_t block_control_stages;
  // SQ_PERFCOUNTER_CTRL2 and SQG_PERFCOUNTER_CTRL2 with their reset default:
  // every VMID, FORCE_EN clear.
  uint32_t instance_block_filter;
  uint32_t engine_block_filter;
  uint32_t block_filter_default;
  // SQG_PERFCOUNTER0_SELECT; select n is 4n above.
  uint32_t engine_select;
  // SQ_PERFCOUNTER0_SELECT; usable select 2k is 8k above.
  uint32_t instance_select;
  // Reset default of every select register.
  uint32_t select_default;
  // SQG_PERFCOUNTER0_LO; counter n is 8n above and its high half 4 above.
  uint32_t engine_result;
  // SQ_PERFCOUNTER0_LO; counter k is 8k above, 32 bits.
  uint32_t instance_result;
  // COMPUTE_PERFCOUNT_ENABLE as an SH register index.
  uint32_t compute_enable;
  // Selectable events.
  const iree_hal_amd_gpu_pm4_perf_counter_event_t* events;
  uint32_t event_count;
} iree_hal_amd_gpu_pm4_perf_counter_profile_t;

// Selects the register table of a qualified physical ISA. Unqualified targets
// return UNIMPLEMENTED and leave the profile zeroed.
iree_status_t iree_hal_amd_gpu_pm4_perf_counter_profile_initialize(
    iree_hal_amd_gpu_gfxip_version_t gfxip,
    iree_hal_amd_gpu_pm4_perf_counter_profile_t* out_profile);

// Finds an event by its exact name, or returns NULL.
const iree_hal_amd_gpu_pm4_perf_counter_event_t*
iree_hal_amd_gpu_pm4_perf_counter_profile_find(
    const iree_hal_amd_gpu_pm4_perf_counter_profile_t* profile,
    iree_string_view_t name);

// One prepared counter selection. Immutable once validated; every sequence
// and every raw sample of a session uses the same one.
typedef struct iree_hal_amd_gpu_pm4_perf_counter_config_t {
  // Borrowed target register table.
  const iree_hal_amd_gpu_pm4_perf_counter_profile_t* profile;
  // Shader engines of the device; each is sampled.
  uint32_t engine_count;
  // Selected engine events in slot order.
  uint32_t engine_counter_count;
  const iree_hal_amd_gpu_pm4_perf_counter_event_t*
      engine_counters[IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_MAX_ENGINE_COUNTERS];
  // Selected instance events in result-register order.
  uint32_t instance_counter_count;
  const iree_hal_amd_gpu_pm4_perf_counter_event_t* instance_counters
      [IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_MAX_INSTANCE_COUNTERS];
  // Graphics-index value of each workgroup processor the device has; empty
  // when no instance counter is selected.
  uint32_t instance_count;
  uint32_t instances[IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_MAX_INSTANCE_COUNT];
} iree_hal_amd_gpu_pm4_perf_counter_config_t;

// Graphics-index value addressing one workgroup processor.
static inline uint32_t iree_hal_amd_gpu_pm4_perf_counter_instance(
    uint32_t shader_engine, uint32_t shader_array,
    uint32_t workgroup_processor) {
  return (shader_engine << 16) | (shader_array << 8) |
         (workgroup_processor << 2);
}

// Checks the counts, that every event belongs to its list's block and that
// instance counters have instances to read.
iree_status_t iree_hal_amd_gpu_pm4_perf_counter_config_validate(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config);

// Identity of a selection's sample layout: equal for two configs exactly when
// their sample sequences and raw samples are interchangeable. Never zero.
uint64_t iree_hal_amd_gpu_pm4_perf_counter_config_signature(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config);

// Values in one sample: one per engine for each engine counter, then one per
// instance for each instance counter. A counter's values are adjacent.
uint32_t iree_hal_amd_gpu_pm4_perf_counter_value_count(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config);

// First value of an engine counter, or of an instance counter, in a sample.
uint32_t iree_hal_amd_gpu_pm4_perf_counter_engine_value_offset(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config, uint32_t counter);
uint32_t iree_hal_amd_gpu_pm4_perf_counter_instance_value_offset(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config, uint32_t counter);

// Byte length of one raw sample: eight bytes per value, DWORD aligned. An
// engine value is written as two DWORDs, an instance value as its low DWORD.
static inline uint32_t iree_hal_amd_gpu_pm4_perf_counter_sample_byte_length(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config) {
  return iree_hal_amd_gpu_pm4_perf_counter_value_count(config) * 8;
}

// Exact DWORD counts of the three sequences for a validated config.
uint32_t iree_hal_amd_gpu_pm4_perf_counter_start_word_count(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config);
uint32_t iree_hal_amd_gpu_pm4_perf_counter_sample_word_count(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config);
uint32_t iree_hal_amd_gpu_pm4_perf_counter_stop_word_count(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config);

// Joins earlier compute work, copies the STATE_WORD_COUNT registers a
// collector would have changed to the DWORD aligned state_address, then
// programs the selection with counting disabled, sets this queue's enable and
// starts every counter from zero. Leaves the graphics index at broadcast.
uint32_t* iree_hal_amd_gpu_pm4_perf_counter_emit_start(
    uint32_t* words, const iree_hal_amd_gpu_pm4_perf_counter_config_t* config,
    uint64_t state_address);

// Joins earlier compute work, latches every counter without stopping it and
// copies each instance to an eight-byte aligned raw sample at target_address.
// The result registers are latches: without the strobe they read zero. Leaves
// the graphics index at broadcast. The sequence begins with a compute join, so
// a recording can disable it by replacing its first DWORD with a NOP header
// covering sample_word_count.
uint32_t* iree_hal_amd_gpu_pm4_perf_counter_emit_sample(
    uint32_t* words, const iree_hal_amd_gpu_pm4_perf_counter_config_t* config,
    uint64_t target_address);

// Copies emitted by one sample sequence.
uint32_t iree_hal_amd_gpu_pm4_perf_counter_sample_copy_count(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config);

// Where copy |index| of a sample sequence stores its destination. A recording
// uses these to relocate a sequence emitted against address zero: the address
// pair occupies two DWORDs at word_offset and names the raw sample's byte at
// byte_offset.
void iree_hal_amd_gpu_pm4_perf_counter_sample_copy(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config, uint32_t index,
    uint32_t* out_word_offset, uint32_t* out_byte_offset);

// Stops counting, clears this queue's enable and returns the control, block
// control, block filter and select registers to their reset defaults. Leaves
// the graphics index at broadcast.
uint32_t* iree_hal_amd_gpu_pm4_perf_counter_emit_stop(
    uint32_t* words, const iree_hal_amd_gpu_pm4_perf_counter_config_t* config);

// Fills raw sample storage so an uncopied DWORD cannot read as a count.
void iree_hal_amd_gpu_pm4_perf_counter_sample_reset(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config, void* sample);

// Subtracts two raw samples into value_count values. Engine values are 64-bit
// differences and instance values 32-bit ones, so a wrapped hardware counter
// still yields the right difference. Returns false, leaving the values zero,
// when a copy of either sample never ran.
bool iree_hal_amd_gpu_pm4_perf_counter_sample_delta(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config, const void* begin,
    const void* end, uint64_t* out_values);

// True when the state words the start sequence copied show the reset defaults
// a collector leaves behind. False means another collector's programming was
// in place, or the words were never copied.
bool iree_hal_amd_gpu_pm4_perf_counter_state_is_default(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config,
    const uint32_t* state_words);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_PERF_COUNTERS_H_
