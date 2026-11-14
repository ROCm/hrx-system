// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_THREAD_TRACE_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_THREAD_TRACE_H_

#include "iree/base/api.h"
#include "iree/hal/driver/amd/gpu/target/identity.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// SQ thread trace (SQTT) bracket for one PM4 compute publication. The start
// sequence programs one finite buffer per traced shader engine and enables the
// trace; the stop sequence finishes it, waits in the ring for the hardware to
// drain, turns the mode off and copies the write pointer, status and dropped
// count of each engine to host-visible memory. Both sequences leave the
// graphics index register at full broadcast.
//
// The trace registers are device-wide, not queue context. At most one bracket
// is open on a device at a time, and a second collector (rocprofv3 ATT, RADV
// thread trace) reprograms the same registers. The data is per process: without
// ALL_VMID the hardware records only the tracing process's waves.
//
// One SIMD of one workgroup processor of array zero is traced per shader
// engine, so a trace samples a small fraction of a dispatch's waves. It is not
// a census.

enum {
  // Shader engines addressable by one trace bracket.
  IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_MAX_ENGINE_COUNT = 8,
  // Per-engine buffer bounds. The upper bound is exclusive on GFX11.
  IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_MIN_BUFFER_LENGTH = 128 * 1024,
  IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_MAX_BUFFER_LENGTH = 512 * 1024 * 1024,
  // Buffer base and length granularity.
  IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_BUFFER_ALIGNMENT = 4096,
  // DWORDs copied back per traced engine: write pointer, status, dropped count
  // and one reserved word.
  IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_INFO_WORD_COUNT = 4,
  // Host-written value of an info word the stop sequence has not replaced.
  IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_INFO_UNWRITTEN = (int)0xffffffff,
};

// Target register table and control values. Register fields are byte
// addresses, except compute_enable which is an SH register index.
typedef struct iree_hal_amd_gpu_pm4_thread_trace_profile_t {
  // GRBM_GFX_INDEX.
  uint32_t graphics_index;
  // SPI_CONFIG_CNTL and its two values. Without the SQG event bits the finish
  // event never reports completion. The idle value is the reset default.
  uint32_t event_control;
  uint32_t event_control_tracing;
  uint32_t event_control_idle;
  // SQ_THREAD_TRACE_BUF0_BASE, in 4 KiB units.
  uint32_t buffer_base;
  // SQ_THREAD_TRACE_BUF0_SIZE: 4 KiB units above bit 8, high base bits below.
  uint32_t buffer_size;
  // SQ_THREAD_TRACE_CTRL and its values with the mode on and off.
  uint32_t control;
  uint32_t control_tracing;
  uint32_t control_idle;
  // SQ_THREAD_TRACE_MASK; workgroup processor above bit 4, SIMD below.
  uint32_t mask;
  uint32_t mask_wave_types;
  // SQ_THREAD_TRACE_TOKEN_MASK and the decodable token selection.
  uint32_t token_mask;
  uint32_t token_mask_value;
  // SQ_THREAD_TRACE_WPTR, in 32-byte units.
  uint32_t write_pointer;
  // SQ_THREAD_TRACE_STATUS and its fields.
  uint32_t status;
  uint32_t status_finish_done_mask;
  uint32_t status_write_error_mask;
  uint32_t status_busy_mask;
  // SQ_THREAD_TRACE_DROPPED_CNTR.
  uint32_t dropped_count;
  // COMPUTE_THREAD_TRACE_ENABLE as an SH register index.
  uint32_t compute_enable;
  // VGT event finishing the trace.
  uint32_t finish_event;
  // Highest selectable workgroup processor and SIMD.
  uint32_t maximum_workgroup_processor;
  uint32_t maximum_simd;
  // The write pointer counts from the buffer base address instead of zero.
  bool write_pointer_includes_base;
} iree_hal_amd_gpu_pm4_thread_trace_profile_t;

// Selects the register table of a qualified physical ISA. Unqualified targets
// return UNIMPLEMENTED and leave the profile zeroed.
iree_status_t iree_hal_amd_gpu_pm4_thread_trace_profile_initialize(
    iree_hal_amd_gpu_gfxip_version_t gfxip,
    iree_hal_amd_gpu_pm4_thread_trace_profile_t* out_profile);

// One prepared bracket. Buffers are caller-owned, GPU-writable and stay mapped
// until the host has read the written extent.
typedef struct iree_hal_amd_gpu_pm4_thread_trace_config_t {
  // Borrowed target register table.
  const iree_hal_amd_gpu_pm4_thread_trace_profile_t* profile;
  // Shader engines of the device.
  uint32_t engine_count;
  // Engines traced; bits at or above engine_count are ignored.
  uint32_t engine_mask;
  // Workgroup processor and SIMD selected on every traced engine.
  uint32_t workgroup_processor;
  uint32_t simd;
  // Byte length of each engine's buffer.
  uint64_t buffer_length;
  // 4 KiB aligned GPU address of each engine's buffer, indexed by engine.
  uint64_t buffer_addresses[IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_MAX_ENGINE_COUNT];
  // DWORD aligned GPU address of the info words, INFO_WORD_COUNT per engine,
  // indexed by engine.
  uint64_t info_address;
} iree_hal_amd_gpu_pm4_thread_trace_config_t;

// Checks buffer bounds and alignment, the engine selection and the selectors.
iree_status_t iree_hal_amd_gpu_pm4_thread_trace_config_validate(
    const iree_hal_amd_gpu_pm4_thread_trace_config_t* config);

// Exact DWORD counts of the two sequences for a validated config.
uint32_t iree_hal_amd_gpu_pm4_thread_trace_start_word_count(
    const iree_hal_amd_gpu_pm4_thread_trace_config_t* config);
uint32_t iree_hal_amd_gpu_pm4_thread_trace_stop_word_count(
    const iree_hal_amd_gpu_pm4_thread_trace_config_t* config);

// Joins earlier compute work, then enables the trace on every selected engine.
// The control register is written last. Work published after this sequence is
// traced until the stop sequence.
uint32_t* iree_hal_amd_gpu_pm4_thread_trace_emit_start(
    uint32_t* words, const iree_hal_amd_gpu_pm4_thread_trace_config_t* config);

// Joins the traced compute work, finishes the trace and copies each engine's
// info words. The register waits hold the queue until the hardware reports the
// finish done and the trace idle, which the event control value guarantees.
uint32_t* iree_hal_amd_gpu_pm4_thread_trace_emit_stop(
    uint32_t* words, const iree_hal_amd_gpu_pm4_thread_trace_config_t* config);

// Host view of one engine's info words after the stop sequence retired.
typedef struct iree_hal_amd_gpu_pm4_thread_trace_result_t {
  // Bytes the hardware wrote from the buffer start; never above buffer_length.
  uint64_t written_length;
  // Raw SQ_THREAD_TRACE_STATUS.
  uint32_t status;
  // Raw SQ_THREAD_TRACE_DROPPED_CNTR.
  uint32_t dropped_count;
  // The stop sequence replaced every info word.
  bool collected;
  // The finish event completed on this engine.
  bool finish_done;
  // The hardware reported a buffer write error.
  bool write_error;
  // The write pointer stopped at the end of the buffer.
  bool buffer_full;
} iree_hal_amd_gpu_pm4_thread_trace_result_t;

// Decodes one engine's info words. An uncollected result has zero length.
void iree_hal_amd_gpu_pm4_thread_trace_read_info(
    const iree_hal_amd_gpu_pm4_thread_trace_config_t* config,
    uint32_t shader_engine, const uint32_t* info_words,
    iree_hal_amd_gpu_pm4_thread_trace_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_THREAD_TRACE_H_
