// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/pm4/thread_trace.h"

#include <inttypes.h>

#include "iree/base/internal/math.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/commands.h"

// Registers and values GFX11.0 and GFX11.5 share. Mesa's gfx11 and gfx115
// register databases give both generations the same addresses and field
// layouts for every register here, and Mesa and aqlprofile drive both through
// one code path. Published and decoded on gfx1151 only.
static const iree_hal_amd_gpu_pm4_thread_trace_profile_t
    iree_hal_amd_gpu_pm4_thread_trace_gfx11_registers = {
        .graphics_index = 0x30800,
        .event_control = 0x31100,
        // SQG top and bottom event enables over the reset default.
        .event_control_tracing = UINT32_C(0xc362c688),
        .event_control_idle = UINT32_C(0xc062c688),
        .buffer_base = 0x367a0,
        .buffer_size = 0x367a4,
        .buffer_size_shift = 8,
        .control = 0x367b0,
        // Draw events, 4096-clock realtime tokens, utilization timer, SQ and
        // SPI stalls instead of token loss, registers at high water, high
        // water 5/8.
        .control_tracing = UINT32_C(0x80023d41),
        .control_idle = UINT32_C(0x80023d40),
        .mask = 0x367b4,
        // PS, GS, HS and CS wave types.
        .mask_wave_types = 0x55u << 10,
        .token_mask = 0x367b8,
        // All register tokens; no VMEM/ALU execution, event,
        // utilization-counter or wave-allocation tokens and no bottom-of-pipe
        // events. The ROCm trace decoder has no encoding for the excluded
        // tokens.
        .token_mask_value = UINT32_C(0x003f0683),
        .write_pointer = 0x367bc,
        .status = 0x367d0,
        .status_finish_done_mask = UINT32_C(0x00fff000),
        .status_write_error_mask = UINT32_C(0x01000000),
        .status_busy_mask = UINT32_C(0x02000000),
        .dropped_count = 0x367e8,
        .compute_enable = 0x2c00 + ((0xb878 - 0xb000) >> 2),
        .finish_event = 55,
        .maximum_workgroup_processor = 15,
        .maximum_simd = 3,
};

// GFX12.0, transcribed from Mesa (ac_sqtt_emit_start, ac_sqtt_get_ctrl,
// ac_emit_cp_spi_config_cntl) and its gfx12 register database. Never run.
static const iree_hal_amd_gpu_pm4_thread_trace_profile_t
    iree_hal_amd_gpu_pm4_thread_trace_gfx12_registers = {
        .graphics_index = 0x30800,
        // SPI_SQG_EVENT_CTL holds only the two event enables.
        .event_control = 0x31120,
        .event_control_tracing = 0x3,
        .event_control_idle = 0x0,
        .buffer_base = 0x3679c,
        .buffer_base_high = 0x367a0,
        .buffer_size = 0x36798,
        .buffer_size_shift = 0,
        .control = 0x367b0,
        // As GFX11 without realtime tokens, which GFX12 has no field for: the
        // utilization timer sits one bit higher and the low-water offset is 4.
        .control_tracing = UINT32_C(0x80405d41),
        .control_idle = UINT32_C(0x80405d40),
        .mask = 0x367b4,
        .mask_wave_types = 0x55u << 10,
        .token_mask = 0x367b8,
        // The GFX11 selection, whose bits keep their positions, plus the two
        // GFX12-only fields as Mesa sets them: barrier waits and the
        // CP_ME_MC_RADDR register class excluded.
        .token_mask_value = UINT32_C(0x083f4683),
        .write_pointer = 0x367bc,
        .status = 0x367d0,
        .status_finish_done_mask = UINT32_C(0x00fff000),
        .status_write_error_mask = UINT32_C(0x01000000),
        .status_busy_mask = UINT32_C(0x02000000),
        .dropped_count = 0x367e8,
        .compute_enable = 0x2c00 + ((0xb878 - 0xb000) >> 2),
        .finish_event = 55,
        .maximum_workgroup_processor = 15,
        .maximum_simd = 3,
};

// One generation of the capability table.
typedef struct iree_hal_amd_gpu_pm4_thread_trace_target_t {
  // gfx IP major and minor version the row applies to.
  uint32_t major;
  uint32_t minor;
  // Steppings whose bracket ran on hardware, one bit per stepping.
  uint32_t verified_steppings;
  // Register table of the generation.
  const iree_hal_amd_gpu_pm4_thread_trace_profile_t* registers;
  // The write pointer counts from the buffer base address instead of zero.
  bool write_pointer_includes_base;
} iree_hal_amd_gpu_pm4_thread_trace_target_t;

// A generation without a table, and what a table for it would need.
typedef struct iree_hal_amd_gpu_pm4_thread_trace_gap_t {
  // gfx IP major version, and its minor or UINT32_MAX for every minor.
  uint32_t major;
  uint32_t minor;
  // What differs and where the missing values are documented.
  const char* reason;
} iree_hal_amd_gpu_pm4_thread_trace_gap_t;

// Only GFX11.5 stepping 1 (gfx1151) has run: no other GPU was available. Every
// other row and stepping is transcribed from references and unverified.
static const iree_hal_amd_gpu_pm4_thread_trace_target_t
    iree_hal_amd_gpu_pm4_thread_trace_targets[] = {
        // Measured on gfx1151: the pointer counts from zero.
        {11, 5, 1u << 1, &iree_hal_amd_gpu_pm4_thread_trace_gfx11_registers,
         false},
        // Mesa (ac_sqtt_copy_info_regs) and aqlprofile (threadtrace.cpp) both
        // subtract the buffer base on GFX11.0 and on no other generation.
        {11, 0, 0, &iree_hal_amd_gpu_pm4_thread_trace_gfx11_registers, true},
        {12, 0, 0, &iree_hal_amd_gpu_pm4_thread_trace_gfx12_registers, false},
};

static const iree_hal_amd_gpu_pm4_thread_trace_gap_t
    iree_hal_amd_gpu_pm4_thread_trace_gaps[] = {
        {9, UINT32_MAX,
         "GFX9 (CDNA) thread trace uses another register set "
         "(SQ_THREAD_TRACE_BASE, BASE2, SIZE, MODE, TOKEN_MASK2, HIWATER) and "
         "a sequence without a finish event, so it needs a second bracket; "
         "the values are in Mesa ac_sqtt.c (GFX9 path) and aqlprofile "
         "gfx9_primitives.h"},
        {10, UINT32_MAX,
         "GFX10 keeps the trace registers in privileged config space "
         "(0x8d00-0x8d3c), written with a different packet, and has "
         "chip-specific stop quirks; the values are in Mesa ac_sqtt.c (GFX10 "
         "path)"},
        {11, 7,
         "no register database for GFX11.7 is available, and the references "
         "disagree on its write pointer: Mesa treats it like GFX11.5 (from "
         "zero), aqlprofile like GFX11.0 (from the buffer base)"},
        {12, 5,
         "GFX12.5 has its own register set (gc_12_1) and per-XCC selection "
         "that the bracket does not model; see aqlprofile gfx12_primitives.h "
         "(GFX12_VARIANT_1250)"},
};

iree_status_t iree_hal_amd_gpu_pm4_thread_trace_profile_initialize(
    iree_hal_amd_gpu_gfxip_version_t gfxip,
    iree_hal_amd_gpu_pm4_thread_trace_profile_t* out_profile) {
  *out_profile = (iree_hal_amd_gpu_pm4_thread_trace_profile_t){0};
  for (uint32_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_amd_gpu_pm4_thread_trace_targets); ++i) {
    const iree_hal_amd_gpu_pm4_thread_trace_target_t* target =
        &iree_hal_amd_gpu_pm4_thread_trace_targets[i];
    if (target->major != gfxip.major || target->minor != gfxip.minor) {
      continue;
    }
    *out_profile = *target->registers;
    out_profile->write_pointer_includes_base =
        target->write_pointer_includes_base;
    out_profile->verified =
        gfxip.stepping < 32 &&
        (target->verified_steppings & (1u << gfxip.stepping)) != 0;
    return iree_ok_status();
  }
  const char* reason = "no register table has been written for this generation";
  for (uint32_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_amd_gpu_pm4_thread_trace_gaps); ++i) {
    const iree_hal_amd_gpu_pm4_thread_trace_gap_t* gap =
        &iree_hal_amd_gpu_pm4_thread_trace_gaps[i];
    if (gap->major == gfxip.major &&
        (gap->minor == UINT32_MAX || gap->minor == gfxip.minor)) {
      reason = gap->reason;
      break;
    }
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "no PM4 thread trace table for gfx%u.%u.%u: %s",
                          gfxip.major, gfxip.minor, gfxip.stepping, reason);
}

static uint32_t iree_hal_amd_gpu_pm4_thread_trace_engines(
    const iree_hal_amd_gpu_pm4_thread_trace_config_t* config) {
  return config->engine_mask & ((1u << config->engine_count) - 1);
}

iree_status_t iree_hal_amd_gpu_pm4_thread_trace_config_validate(
    const iree_hal_amd_gpu_pm4_thread_trace_config_t* config) {
  const iree_hal_amd_gpu_pm4_thread_trace_profile_t* profile = config->profile;
  if (!config->engine_count ||
      config->engine_count >
          IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_MAX_ENGINE_COUNT ||
      !iree_hal_amd_gpu_pm4_thread_trace_engines(config)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "thread trace selects no shader engine: mask 0x%x of %u engines",
        config->engine_mask, config->engine_count);
  }
  if (config->workgroup_processor > profile->maximum_workgroup_processor ||
      config->simd > profile->maximum_simd) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "thread trace selector is out of range: workgroup processor %u (at "
        "most %u), SIMD %u (at most %u)",
        config->workgroup_processor, profile->maximum_workgroup_processor,
        config->simd, profile->maximum_simd);
  }
  if (config->buffer_length <
          IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_MIN_BUFFER_LENGTH ||
      config->buffer_length >=
          IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_MAX_BUFFER_LENGTH ||
      config->buffer_length %
          IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_BUFFER_ALIGNMENT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "thread trace buffer length %" PRIu64
        " must be a 4 KiB multiple of at least 128 KiB and below 512 MiB",
        config->buffer_length);
  }
  if (config->info_address & 3) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "thread trace info address is misaligned");
  }
  for (uint32_t i = 0; i < config->engine_count; ++i) {
    if (!(iree_hal_amd_gpu_pm4_thread_trace_engines(config) & (1u << i))) {
      continue;
    }
    const uint64_t address = config->buffer_addresses[i];
    // The base register holds 32 page-number bits and the size register four.
    if (!address ||
        address % IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_BUFFER_ALIGNMENT ||
        (address + config->buffer_length) >> 48) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "thread trace buffer %u address 0x%" PRIx64
                              " is not programmable",
                              i, address);
    }
  }
  return iree_ok_status();
}

uint32_t iree_hal_amd_gpu_pm4_thread_trace_start_word_count(
    const iree_hal_amd_gpu_pm4_thread_trace_config_t* config) {
  // A separate high base register adds its own write and a pointer reset.
  const uint32_t engine_words = config->profile->buffer_base_high ? 24 : 18;
  return 11 +
         engine_words * (uint32_t)iree_math_count_ones_u32(
                            iree_hal_amd_gpu_pm4_thread_trace_engines(config));
}

uint32_t iree_hal_amd_gpu_pm4_thread_trace_stop_word_count(
    const iree_hal_amd_gpu_pm4_thread_trace_config_t* config) {
  return 13 + 38 * (uint32_t)iree_math_count_ones_u32(
                       iree_hal_amd_gpu_pm4_thread_trace_engines(config));
}

static uint32_t* iree_hal_amd_gpu_pm4_thread_trace_emit_register(
    uint32_t* words, uint32_t register_address, uint32_t value) {
  return iree_hal_amd_gpu_pm4_emit_uconfig_registers(words, register_address, 1,
                                                     &value);
}

uint32_t* iree_hal_amd_gpu_pm4_thread_trace_emit_start(
    uint32_t* words, const iree_hal_amd_gpu_pm4_thread_trace_config_t* config) {
  const iree_hal_amd_gpu_pm4_thread_trace_profile_t* profile = config->profile;
  const uint32_t engines = iree_hal_amd_gpu_pm4_thread_trace_engines(config);
  // Earlier work on this queue must not appear in the trace.
  words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
  words = iree_hal_amd_gpu_pm4_thread_trace_emit_register(
      words, profile->event_control, profile->event_control_tracing);
  for (uint32_t i = 0; i < config->engine_count; ++i) {
    if (!(engines & (1u << i))) {
      continue;
    }
    const uint64_t page = config->buffer_addresses[i] >> 12;
    words = iree_hal_amd_gpu_pm4_emit_engine_select(words,
                                                    profile->graphics_index, i);
    // The size precedes the base. Without a high base register the size
    // register carries the high base bits below the size.
    words = iree_hal_amd_gpu_pm4_thread_trace_emit_register(
        words, profile->buffer_size,
        ((uint32_t)(config->buffer_length >> 12)
         << profile->buffer_size_shift) |
            (profile->buffer_base_high ? 0 : (uint32_t)(page >> 32)));
    words = iree_hal_amd_gpu_pm4_thread_trace_emit_register(
        words, profile->buffer_base, (uint32_t)page);
    if (profile->buffer_base_high) {
      words = iree_hal_amd_gpu_pm4_thread_trace_emit_register(
          words, profile->buffer_base_high, (uint32_t)(page >> 32));
      // The pointer is not reset by programming the buffer.
      words = iree_hal_amd_gpu_pm4_thread_trace_emit_register(
          words, profile->write_pointer, 0);
    }
    words = iree_hal_amd_gpu_pm4_thread_trace_emit_register(
        words, profile->mask,
        profile->mask_wave_types | (config->workgroup_processor << 4) |
            config->simd);
    words = iree_hal_amd_gpu_pm4_thread_trace_emit_register(
        words, profile->token_mask, profile->token_mask_value);
    // Last: the mode bit starts the trace on this engine.
    words = iree_hal_amd_gpu_pm4_thread_trace_emit_register(
        words, profile->control, profile->control_tracing);
  }
  words = iree_hal_amd_gpu_pm4_emit_engine_broadcast(words,
                                                     profile->graphics_index);
  const uint32_t enable = 1;
  return iree_hal_amd_gpu_pm4_emit_registers(words, profile->compute_enable, 1,
                                             &enable);
}

uint32_t* iree_hal_amd_gpu_pm4_thread_trace_emit_stop(
    uint32_t* words, const iree_hal_amd_gpu_pm4_thread_trace_config_t* config) {
  const iree_hal_amd_gpu_pm4_thread_trace_profile_t* profile = config->profile;
  const uint32_t engines = iree_hal_amd_gpu_pm4_thread_trace_engines(config);
  // Every traced wave has left the shader core before the finish event.
  words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
  const uint32_t disable = 0;
  words = iree_hal_amd_gpu_pm4_emit_registers(words, profile->compute_enable, 1,
                                              &disable);
  words = iree_hal_amd_gpu_pm4_emit_event(words, profile->finish_event);
  for (uint32_t i = 0; i < config->engine_count; ++i) {
    if (!(engines & (1u << i))) {
      continue;
    }
    const uint64_t info =
        config->info_address +
        (uint64_t)i * IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_INFO_WORD_COUNT * 4;
    words = iree_hal_amd_gpu_pm4_emit_engine_select(words,
                                                    profile->graphics_index, i);
    words = iree_hal_amd_gpu_pm4_emit_wait_register32(
        words, profile->status, 0, profile->status_finish_done_mask,
        IREE_HAL_AMD_GPU_PM4_COMPARISON_NOT_EQUAL);
    words = iree_hal_amd_gpu_pm4_thread_trace_emit_register(
        words, profile->control, profile->control_idle);
    words = iree_hal_amd_gpu_pm4_emit_wait_register32(
        words, profile->status, 0, profile->status_busy_mask,
        IREE_HAL_AMD_GPU_PM4_COMPARISON_EQUAL);
    words = iree_hal_amd_gpu_pm4_emit_copy_register32(
        words, profile->write_pointer, info);
    words = iree_hal_amd_gpu_pm4_emit_copy_register32(words, profile->status,
                                                      info + 4);
    words = iree_hal_amd_gpu_pm4_emit_copy_register32(
        words, profile->dropped_count, info + 8);
  }
  words = iree_hal_amd_gpu_pm4_emit_engine_broadcast(words,
                                                     profile->graphics_index);
  return iree_hal_amd_gpu_pm4_thread_trace_emit_register(
      words, profile->event_control, profile->event_control_idle);
}

void iree_hal_amd_gpu_pm4_thread_trace_read_info(
    const iree_hal_amd_gpu_pm4_thread_trace_config_t* config,
    uint32_t shader_engine, const uint32_t* info_words,
    iree_hal_amd_gpu_pm4_thread_trace_result_t* out_result) {
  const iree_hal_amd_gpu_pm4_thread_trace_profile_t* profile = config->profile;
  *out_result = (iree_hal_amd_gpu_pm4_thread_trace_result_t){0};
  const uint32_t unwritten =
      (uint32_t)IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_INFO_UNWRITTEN;
  const uint32_t write_pointer = info_words[0];
  const uint32_t status = info_words[1];
  const uint32_t dropped_count = info_words[2];
  // The dropped count and status can legitimately hold any value; only all
  // three words still at the host pattern mean the copies never ran.
  if (write_pointer == unwritten && status == unwritten &&
      dropped_count == unwritten) {
    return;
  }
  out_result->collected = true;
  out_result->status = status;
  out_result->dropped_count = dropped_count;
  out_result->finish_done = (status & profile->status_finish_done_mask) != 0;
  out_result->write_error = (status & profile->status_write_error_mask) != 0;
  uint32_t offset = write_pointer & UINT32_C(0x1fffffff);
  if (profile->write_pointer_includes_base) {
    offset =
        (offset - (uint32_t)(config->buffer_addresses[shader_engine] >> 5)) &
        UINT32_C(0x1fffffff);
  }
  const uint64_t written = (uint64_t)offset * 32;
  // The pointer stops one 32-byte unit short of a full buffer.
  out_result->buffer_full = written + 32 >= config->buffer_length;
  out_result->written_length = iree_min(written, config->buffer_length);
}
