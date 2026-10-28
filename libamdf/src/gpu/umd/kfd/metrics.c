// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/metrics.h"

#include <errno.h>
#include <string.h>

// Native layout and producer facts: Linux kgd_pp_interface.h, smu_cmn.h,
// and the SMU11--SMU15 get_gpu_metrics producers at
// f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931. Tables record only interpreted
// fields, not copies of the complete firmware/native structures.

enum amdf_gpu_kfd_metrics_field_e {
  AMDF_METRICS_TEMPERATURE_EDGE,
  AMDF_METRICS_TEMPERATURE_HOTSPOT,
  AMDF_METRICS_TEMPERATURE_MEM,
  AMDF_METRICS_TEMPERATURE_GFX,
  AMDF_METRICS_TEMPERATURE_SOC,
  AMDF_METRICS_TEMPERATURE_CORE,
  AMDF_METRICS_TEMPERATURE_HBM,
  AMDF_METRICS_TEMPERATURE_SKIN,
  AMDF_METRICS_AVERAGE_GFX_ACTIVITY,
  AMDF_METRICS_AVERAGE_UMC_ACTIVITY,
  AMDF_METRICS_AVERAGE_IPU_ACTIVITY,
  AMDF_METRICS_AVERAGE_CORE_C0_ACTIVITY,
  AMDF_METRICS_AVERAGE_SOCKET_POWER,
  AMDF_METRICS_CURR_SOCKET_POWER,
  AMDF_METRICS_AVERAGE_CPU_POWER,
  AMDF_METRICS_AVERAGE_SOC_POWER,
  AMDF_METRICS_AVERAGE_GFX_POWER,
  AMDF_METRICS_AVERAGE_IPU_POWER,
  AMDF_METRICS_AVERAGE_APU_POWER,
  AMDF_METRICS_AVERAGE_ALL_CORE_POWER,
  AMDF_METRICS_AVERAGE_CORE_POWER,
  AMDF_METRICS_AVERAGE_SYS_POWER,
  AMDF_METRICS_ENERGY_ACCUMULATOR,
  AMDF_METRICS_AVERAGE_GFXCLK_FREQUENCY,
  AMDF_METRICS_AVERAGE_SOCCLK_FREQUENCY,
  AMDF_METRICS_AVERAGE_UCLK_FREQUENCY,
  AMDF_METRICS_AVERAGE_FCLK_FREQUENCY,
  AMDF_METRICS_CURRENT_GFXCLK,
  AMDF_METRICS_CURRENT_SOCCLK,
  AMDF_METRICS_CURRENT_UCLK,
  AMDF_METRICS_CURRENT_FCLK,
  AMDF_METRICS_CURRENT_CORECLK,
  AMDF_METRICS_AVERAGE_DRAM_READS,
  AMDF_METRICS_AVERAGE_DRAM_WRITES,
  AMDF_METRICS_AVERAGE_IPU_READS,
  AMDF_METRICS_AVERAGE_IPU_WRITES,
  AMDF_METRICS_AVERAGE_TEMPERATURE_GFX,
  AMDF_METRICS_AVERAGE_TEMPERATURE_SOC,
  AMDF_METRICS_AVERAGE_TEMPERATURE_CORE,
  AMDF_METRICS_SYSTEM_CLOCK_COUNTER,
  AMDF_METRICS_FIRMWARE_TIMESTAMP,
  AMDF_METRICS_TIME_FILTER_ALPHAVALUE,
  AMDF_METRICS_FIELD_COUNT,
};

_Static_assert(AMDF_METRICS_FIELD_COUNT == AMDF_GPU_KFD_METRICS_FIELD_COUNT,
               "snapshot storage matches the interpreted field set");

typedef struct amdf_gpu_kfd_metric_definition_t {
  // Native member name, also used for diagnostics and provenance.
  const char* name;
  // Portable metric meaning; zero denotes sample metadata instead of a value.
  uint8_t kind;
  // Physical domain before a producer-specific interpretation refinement.
  uint8_t domain;
  // Native tagged attribute ID, or UINT16_MAX when no tagged form is
  // established.
  uint16_t tag_id;
  // Native tagged unit ID; producer semantics can refine an unspecified unit.
  uint8_t tag_unit;
  // Native tagged scalar type ID (unsigned width), when a tag is established.
  uint8_t tag_type;
} amdf_gpu_kfd_metric_definition_t;

static const amdf_gpu_kfd_metric_definition_t
    amdf_gpu_kfd_metric_definitions[] = {
        {"temperature_edge", AMDF_TELEMETRY_METRIC_TEMPERATURE_EDGE,
         AMDF_TELEMETRY_DOMAIN_GPU, UINT16_MAX, 0, 0},
        {"temperature_hotspot", AMDF_TELEMETRY_METRIC_TEMPERATURE_HOTSPOT,
         AMDF_TELEMETRY_DOMAIN_GPU, 0, 2, 2},
        {"temperature_mem", AMDF_TELEMETRY_METRIC_TEMPERATURE_MEMORY,
         AMDF_TELEMETRY_DOMAIN_MEMORY, 1, 2, 2},
        {"temperature_gfx", AMDF_TELEMETRY_METRIC_TEMPERATURE,
         AMDF_TELEMETRY_DOMAIN_GPU, UINT16_MAX, 0, 0},
        {"temperature_soc", AMDF_TELEMETRY_METRIC_TEMPERATURE,
         AMDF_TELEMETRY_DOMAIN_SOC, UINT16_MAX, 0, 0},
        {"temperature_core", AMDF_TELEMETRY_METRIC_TEMPERATURE,
         AMDF_TELEMETRY_DOMAIN_CPU_CORE, UINT16_MAX, 0, 0},
        {"temperature_hbm", AMDF_TELEMETRY_METRIC_TEMPERATURE_MEMORY,
         AMDF_TELEMETRY_DOMAIN_MEMORY, 48, 2, 2},
        {"temperature_skin", AMDF_TELEMETRY_METRIC_TEMPERATURE,
         AMDF_TELEMETRY_DOMAIN_SYSTEM, UINT16_MAX, 0, 0},
        {"average_gfx_activity", AMDF_TELEMETRY_METRIC_ACTIVITY_COMPUTE,
         AMDF_TELEMETRY_DOMAIN_GPU, 4, 10, 2},
        {"average_umc_activity", AMDF_TELEMETRY_METRIC_ACTIVITY_MEMORY,
         AMDF_TELEMETRY_DOMAIN_MEMORY_CONTROLLER, 5, 10, 2},
        {"average_ipu_activity", AMDF_TELEMETRY_METRIC_ACTIVITY_COMPUTE,
         AMDF_TELEMETRY_DOMAIN_NPU_COLUMN, UINT16_MAX, 0, 0},
        {"average_core_c0_activity", AMDF_TELEMETRY_METRIC_ACTIVITY_ACTIVE,
         AMDF_TELEMETRY_DOMAIN_CPU_CORE, UINT16_MAX, 0, 0},
        {"average_socket_power", AMDF_TELEMETRY_METRIC_POWER,
         AMDF_TELEMETRY_DOMAIN_PACKAGE, UINT16_MAX, 0, 0},
        {"curr_socket_power", AMDF_TELEMETRY_METRIC_POWER,
         AMDF_TELEMETRY_DOMAIN_PACKAGE, 3, 3, 2},
        {"average_cpu_power", AMDF_TELEMETRY_METRIC_POWER,
         AMDF_TELEMETRY_DOMAIN_CPU, UINT16_MAX, 0, 0},
        {"average_soc_power", AMDF_TELEMETRY_METRIC_POWER,
         AMDF_TELEMETRY_DOMAIN_SOC, UINT16_MAX, 0, 0},
        {"average_gfx_power", AMDF_TELEMETRY_METRIC_POWER,
         AMDF_TELEMETRY_DOMAIN_GPU, UINT16_MAX, 0, 0},
        {"average_ipu_power", AMDF_TELEMETRY_METRIC_POWER,
         AMDF_TELEMETRY_DOMAIN_NPU, UINT16_MAX, 0, 0},
        {"average_apu_power", AMDF_TELEMETRY_METRIC_POWER,
         AMDF_TELEMETRY_DOMAIN_PACKAGE, UINT16_MAX, 0, 0},
        {"average_all_core_power", AMDF_TELEMETRY_METRIC_POWER,
         AMDF_TELEMETRY_DOMAIN_CPU, UINT16_MAX, 0, 0},
        {"average_core_power", AMDF_TELEMETRY_METRIC_POWER,
         AMDF_TELEMETRY_DOMAIN_CPU_CORE, UINT16_MAX, 0, 0},
        {"average_sys_power", AMDF_TELEMETRY_METRIC_POWER,
         AMDF_TELEMETRY_DOMAIN_SYSTEM, UINT16_MAX, 0, 0},
        {"energy_accumulator", AMDF_TELEMETRY_METRIC_ENERGY,
         AMDF_TELEMETRY_DOMAIN_PACKAGE, 7, 0, 6},
        {"average_gfxclk_frequency", AMDF_TELEMETRY_METRIC_CLOCK_COMPUTE,
         AMDF_TELEMETRY_DOMAIN_GPU, UINT16_MAX, 0, 0},
        {"average_socclk_frequency", AMDF_TELEMETRY_METRIC_CLOCK_SOC,
         AMDF_TELEMETRY_DOMAIN_SOC, UINT16_MAX, 0, 0},
        {"average_uclk_frequency", AMDF_TELEMETRY_METRIC_CLOCK_MEMORY,
         AMDF_TELEMETRY_DOMAIN_MEMORY, UINT16_MAX, 0, 0},
        {"average_fclk_frequency", AMDF_TELEMETRY_METRIC_CLOCK_FABRIC,
         AMDF_TELEMETRY_DOMAIN_SOC, UINT16_MAX, 0, 0},
        {"current_gfxclk", AMDF_TELEMETRY_METRIC_CLOCK_COMPUTE,
         AMDF_TELEMETRY_DOMAIN_GPU, 33, 1, 2},
        {"current_socclk", AMDF_TELEMETRY_METRIC_CLOCK_SOC,
         AMDF_TELEMETRY_DOMAIN_SOC, 34, 1, 2},
        {"current_uclk", AMDF_TELEMETRY_METRIC_CLOCK_MEMORY,
         AMDF_TELEMETRY_DOMAIN_MEMORY, 37, 1, 2},
        {"current_fclk", AMDF_TELEMETRY_METRIC_CLOCK_FABRIC,
         AMDF_TELEMETRY_DOMAIN_SOC, UINT16_MAX, 0, 0},
        {"current_coreclk", AMDF_TELEMETRY_METRIC_CLOCK_COMPUTE,
         AMDF_TELEMETRY_DOMAIN_CPU_CORE, UINT16_MAX, 0, 0},
        {"average_dram_reads", AMDF_TELEMETRY_METRIC_TRAFFIC_READ,
         AMDF_TELEMETRY_DOMAIN_MEMORY_CONTROLLER, UINT16_MAX, 0, 0},
        {"average_dram_writes", AMDF_TELEMETRY_METRIC_TRAFFIC_WRITE,
         AMDF_TELEMETRY_DOMAIN_MEMORY_CONTROLLER, UINT16_MAX, 0, 0},
        {"average_ipu_reads", AMDF_TELEMETRY_METRIC_TRAFFIC_READ,
         AMDF_TELEMETRY_DOMAIN_NPU, UINT16_MAX, 0, 0},
        {"average_ipu_writes", AMDF_TELEMETRY_METRIC_TRAFFIC_WRITE,
         AMDF_TELEMETRY_DOMAIN_NPU, UINT16_MAX, 0, 0},
        {"average_temperature_gfx", AMDF_TELEMETRY_METRIC_TEMPERATURE,
         AMDF_TELEMETRY_DOMAIN_GPU, UINT16_MAX, 0, 0},
        {"average_temperature_soc", AMDF_TELEMETRY_METRIC_TEMPERATURE,
         AMDF_TELEMETRY_DOMAIN_SOC, UINT16_MAX, 0, 0},
        {"average_temperature_core", AMDF_TELEMETRY_METRIC_TEMPERATURE,
         AMDF_TELEMETRY_DOMAIN_CPU_CORE, UINT16_MAX, 0, 0},
        {"system_clock_counter", 0, 0, 8, 4, 6},
        {"firmware_timestamp", 0, 0, 32, 5, 6},
        {"time_filter_alphavalue", 0, 0, UINT16_MAX, 0, 0},
};

static const amdf_gpu_kfd_metrics_field_t
    amdf_gpu_kfd_metrics_layout_1_0[AMDF_METRICS_FIELD_COUNT] = {
        [AMDF_METRICS_TEMPERATURE_EDGE] = {16, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_HOTSPOT] = {18, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_MEM] = {20, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_GFX_ACTIVITY] = {28, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_UMC_ACTIVITY] = {30, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_SOCKET_POWER] = {34, 1, 2, 0},
        [AMDF_METRICS_ENERGY_ACCUMULATOR] = {36, 1, 4, 0},
        [AMDF_METRICS_AVERAGE_GFXCLK_FREQUENCY] = {40, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_SOCCLK_FREQUENCY] = {42, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_UCLK_FREQUENCY] = {44, 1, 2, 0},
        [AMDF_METRICS_CURRENT_GFXCLK] = {54, 1, 2, 0},
        [AMDF_METRICS_CURRENT_SOCCLK] = {56, 1, 2, 0},
        [AMDF_METRICS_CURRENT_UCLK] = {58, 1, 2, 0},
        [AMDF_METRICS_SYSTEM_CLOCK_COUNTER] = {8, 1, 8, 0},
};

static const amdf_gpu_kfd_metrics_field_t
    amdf_gpu_kfd_metrics_layout_1_1[AMDF_METRICS_FIELD_COUNT] = {
        [AMDF_METRICS_TEMPERATURE_EDGE] = {4, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_HOTSPOT] = {6, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_MEM] = {8, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_HBM] = {88, 4, 2, 1},
        [AMDF_METRICS_AVERAGE_GFX_ACTIVITY] = {16, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_UMC_ACTIVITY] = {18, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_SOCKET_POWER] = {22, 1, 2, 0},
        [AMDF_METRICS_ENERGY_ACCUMULATOR] = {24, 1, 8, 0},
        [AMDF_METRICS_AVERAGE_GFXCLK_FREQUENCY] = {40, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_SOCCLK_FREQUENCY] = {42, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_UCLK_FREQUENCY] = {44, 1, 2, 0},
        [AMDF_METRICS_CURRENT_GFXCLK] = {54, 1, 2, 0},
        [AMDF_METRICS_CURRENT_SOCCLK] = {56, 1, 2, 0},
        [AMDF_METRICS_CURRENT_UCLK] = {58, 1, 2, 0},
        [AMDF_METRICS_SYSTEM_CLOCK_COUNTER] = {32, 1, 8, 0},
};

static const amdf_gpu_kfd_metrics_field_t
    amdf_gpu_kfd_metrics_layout_1_2[AMDF_METRICS_FIELD_COUNT] = {
        [AMDF_METRICS_TEMPERATURE_EDGE] = {4, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_HOTSPOT] = {6, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_MEM] = {8, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_HBM] = {88, 4, 2, 1},
        [AMDF_METRICS_AVERAGE_GFX_ACTIVITY] = {16, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_UMC_ACTIVITY] = {18, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_SOCKET_POWER] = {22, 1, 2, 0},
        [AMDF_METRICS_ENERGY_ACCUMULATOR] = {24, 1, 8, 0},
        [AMDF_METRICS_AVERAGE_GFXCLK_FREQUENCY] = {40, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_SOCCLK_FREQUENCY] = {42, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_UCLK_FREQUENCY] = {44, 1, 2, 0},
        [AMDF_METRICS_CURRENT_GFXCLK] = {54, 1, 2, 0},
        [AMDF_METRICS_CURRENT_SOCCLK] = {56, 1, 2, 0},
        [AMDF_METRICS_CURRENT_UCLK] = {58, 1, 2, 0},
        [AMDF_METRICS_SYSTEM_CLOCK_COUNTER] = {32, 1, 8, 0},
        [AMDF_METRICS_FIRMWARE_TIMESTAMP] = {96, 1, 8, 0},
};

static const amdf_gpu_kfd_metrics_field_t
    amdf_gpu_kfd_metrics_layout_1_4[AMDF_METRICS_FIELD_COUNT] = {
        [AMDF_METRICS_TEMPERATURE_HOTSPOT] = {4, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_MEM] = {6, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_GFX_ACTIVITY] = {12, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_UMC_ACTIVITY] = {14, 1, 2, 0},
        [AMDF_METRICS_CURR_SOCKET_POWER] = {10, 1, 2, 0},
        [AMDF_METRICS_ENERGY_ACCUMULATOR] = {24, 1, 8, 0},
        [AMDF_METRICS_CURRENT_GFXCLK] = {240, 8, 2, 1},
        [AMDF_METRICS_CURRENT_SOCCLK] = {256, 4, 2, 1},
        [AMDF_METRICS_CURRENT_UCLK] = {280, 1, 2, 0},
        [AMDF_METRICS_SYSTEM_CLOCK_COUNTER] = {32, 1, 8, 0},
        [AMDF_METRICS_FIRMWARE_TIMESTAMP] = {232, 1, 8, 0},
};

static const amdf_gpu_kfd_metrics_field_t
    amdf_gpu_kfd_metrics_layout_1_5[AMDF_METRICS_FIELD_COUNT] = {
        [AMDF_METRICS_TEMPERATURE_HOTSPOT] = {4, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_MEM] = {6, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_GFX_ACTIVITY] = {12, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_UMC_ACTIVITY] = {14, 1, 2, 0},
        [AMDF_METRICS_CURR_SOCKET_POWER] = {10, 1, 2, 0},
        [AMDF_METRICS_ENERGY_ACCUMULATOR] = {88, 1, 8, 0},
        [AMDF_METRICS_CURRENT_GFXCLK] = {312, 8, 2, 1},
        [AMDF_METRICS_CURRENT_SOCCLK] = {328, 4, 2, 1},
        [AMDF_METRICS_CURRENT_UCLK] = {352, 1, 2, 0},
        [AMDF_METRICS_SYSTEM_CLOCK_COUNTER] = {96, 1, 8, 0},
        [AMDF_METRICS_FIRMWARE_TIMESTAMP] = {304, 1, 8, 0},
};

static const amdf_gpu_kfd_metrics_field_t
    amdf_gpu_kfd_metrics_layout_1_6[AMDF_METRICS_FIELD_COUNT] = {
        [AMDF_METRICS_TEMPERATURE_HOTSPOT] = {4, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_MEM] = {6, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_GFX_ACTIVITY] = {12, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_UMC_ACTIVITY] = {14, 1, 2, 0},
        [AMDF_METRICS_CURR_SOCKET_POWER] = {10, 1, 2, 0},
        [AMDF_METRICS_ENERGY_ACCUMULATOR] = {16, 1, 8, 0},
        [AMDF_METRICS_CURRENT_GFXCLK] = {264, 8, 2, 1},
        [AMDF_METRICS_CURRENT_SOCCLK] = {280, 4, 2, 1},
        [AMDF_METRICS_CURRENT_UCLK] = {304, 1, 2, 0},
        [AMDF_METRICS_SYSTEM_CLOCK_COUNTER] = {24, 1, 8, 0},
        [AMDF_METRICS_FIRMWARE_TIMESTAMP] = {256, 1, 8, 0},
};

static const amdf_gpu_kfd_metrics_field_t
    amdf_gpu_kfd_metrics_layout_1_7[AMDF_METRICS_FIELD_COUNT] = {
        [AMDF_METRICS_TEMPERATURE_HOTSPOT] = {4, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_MEM] = {6, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_GFX_ACTIVITY] = {12, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_UMC_ACTIVITY] = {14, 1, 2, 0},
        [AMDF_METRICS_CURR_SOCKET_POWER] = {10, 1, 2, 0},
        [AMDF_METRICS_ENERGY_ACCUMULATOR] = {24, 1, 8, 0},
        [AMDF_METRICS_CURRENT_GFXCLK] = {296, 8, 2, 1},
        [AMDF_METRICS_CURRENT_SOCCLK] = {312, 4, 2, 1},
        [AMDF_METRICS_CURRENT_UCLK] = {336, 1, 2, 0},
        [AMDF_METRICS_SYSTEM_CLOCK_COUNTER] = {32, 1, 8, 0},
        [AMDF_METRICS_FIRMWARE_TIMESTAMP] = {288, 1, 8, 0},
};

static const amdf_gpu_kfd_metrics_field_t
    amdf_gpu_kfd_metrics_layout_2_0[AMDF_METRICS_FIELD_COUNT] = {
        [AMDF_METRICS_TEMPERATURE_GFX] = {16, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_SOC] = {18, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_CORE] = {20, 8, 2, 1},
        [AMDF_METRICS_AVERAGE_GFX_ACTIVITY] = {40, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_SOCKET_POWER] = {44, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_CPU_POWER] = {46, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_SOC_POWER] = {48, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_GFX_POWER] = {50, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_CORE_POWER] = {52, 8, 2, 1},
        [AMDF_METRICS_AVERAGE_GFXCLK_FREQUENCY] = {68, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_SOCCLK_FREQUENCY] = {70, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_UCLK_FREQUENCY] = {72, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_FCLK_FREQUENCY] = {74, 1, 2, 0},
        [AMDF_METRICS_CURRENT_GFXCLK] = {80, 1, 2, 0},
        [AMDF_METRICS_CURRENT_SOCCLK] = {82, 1, 2, 0},
        [AMDF_METRICS_CURRENT_UCLK] = {84, 1, 2, 0},
        [AMDF_METRICS_CURRENT_FCLK] = {86, 1, 2, 0},
        [AMDF_METRICS_CURRENT_CORECLK] = {92, 8, 2, 1},
        [AMDF_METRICS_SYSTEM_CLOCK_COUNTER] = {8, 1, 8, 0},
};

static const amdf_gpu_kfd_metrics_field_t
    amdf_gpu_kfd_metrics_layout_2_1[AMDF_METRICS_FIELD_COUNT] = {
        [AMDF_METRICS_TEMPERATURE_GFX] = {4, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_SOC] = {6, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_CORE] = {8, 8, 2, 1},
        [AMDF_METRICS_AVERAGE_GFX_ACTIVITY] = {28, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_SOCKET_POWER] = {40, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_CPU_POWER] = {42, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_SOC_POWER] = {44, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_GFX_POWER] = {46, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_CORE_POWER] = {48, 8, 2, 1},
        [AMDF_METRICS_AVERAGE_GFXCLK_FREQUENCY] = {64, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_SOCCLK_FREQUENCY] = {66, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_UCLK_FREQUENCY] = {68, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_FCLK_FREQUENCY] = {70, 1, 2, 0},
        [AMDF_METRICS_CURRENT_GFXCLK] = {76, 1, 2, 0},
        [AMDF_METRICS_CURRENT_SOCCLK] = {78, 1, 2, 0},
        [AMDF_METRICS_CURRENT_UCLK] = {80, 1, 2, 0},
        [AMDF_METRICS_CURRENT_FCLK] = {82, 1, 2, 0},
        [AMDF_METRICS_CURRENT_CORECLK] = {88, 8, 2, 1},
        [AMDF_METRICS_SYSTEM_CLOCK_COUNTER] = {32, 1, 8, 0},
};

static const amdf_gpu_kfd_metrics_field_t
    amdf_gpu_kfd_metrics_layout_2_3[AMDF_METRICS_FIELD_COUNT] = {
        [AMDF_METRICS_TEMPERATURE_GFX] = {4, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_SOC] = {6, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_CORE] = {8, 8, 2, 1},
        [AMDF_METRICS_AVERAGE_GFX_ACTIVITY] = {28, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_SOCKET_POWER] = {40, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_CPU_POWER] = {42, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_SOC_POWER] = {44, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_GFX_POWER] = {46, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_CORE_POWER] = {48, 8, 2, 1},
        [AMDF_METRICS_AVERAGE_GFXCLK_FREQUENCY] = {64, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_SOCCLK_FREQUENCY] = {66, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_UCLK_FREQUENCY] = {68, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_FCLK_FREQUENCY] = {70, 1, 2, 0},
        [AMDF_METRICS_CURRENT_GFXCLK] = {76, 1, 2, 0},
        [AMDF_METRICS_CURRENT_SOCCLK] = {78, 1, 2, 0},
        [AMDF_METRICS_CURRENT_UCLK] = {80, 1, 2, 0},
        [AMDF_METRICS_CURRENT_FCLK] = {82, 1, 2, 0},
        [AMDF_METRICS_CURRENT_CORECLK] = {88, 8, 2, 1},
        [AMDF_METRICS_AVERAGE_TEMPERATURE_GFX] = {128, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_TEMPERATURE_SOC] = {130, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_TEMPERATURE_CORE] = {132, 8, 2, 1},
        [AMDF_METRICS_SYSTEM_CLOCK_COUNTER] = {32, 1, 8, 0},
};

static const amdf_gpu_kfd_metrics_field_t
    amdf_gpu_kfd_metrics_layout_3_0[AMDF_METRICS_FIELD_COUNT] = {
        [AMDF_METRICS_TEMPERATURE_GFX] = {4, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_SOC] = {6, 1, 2, 0},
        [AMDF_METRICS_TEMPERATURE_CORE] = {8, 16, 2, 1},
        [AMDF_METRICS_TEMPERATURE_SKIN] = {40, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_GFX_ACTIVITY] = {42, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_IPU_ACTIVITY] = {46, 8, 2, 1},
        [AMDF_METRICS_AVERAGE_CORE_C0_ACTIVITY] = {62, 16, 2, 1},
        [AMDF_METRICS_AVERAGE_SOCKET_POWER] = {112, 1, 4, 0},
        [AMDF_METRICS_AVERAGE_GFX_POWER] = {124, 1, 4, 0},
        [AMDF_METRICS_AVERAGE_IPU_POWER] = {116, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_APU_POWER] = {120, 1, 4, 0},
        [AMDF_METRICS_AVERAGE_ALL_CORE_POWER] = {132, 1, 4, 0},
        [AMDF_METRICS_AVERAGE_CORE_POWER] = {136, 16, 2, 1},
        [AMDF_METRICS_AVERAGE_SYS_POWER] = {168, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_GFXCLK_FREQUENCY] = {174, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_SOCCLK_FREQUENCY] = {176, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_UCLK_FREQUENCY] = {186, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_FCLK_FREQUENCY] = {182, 1, 2, 0},
        [AMDF_METRICS_CURRENT_CORECLK] = {190, 16, 2, 1},
        [AMDF_METRICS_AVERAGE_DRAM_READS] = {94, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_DRAM_WRITES] = {96, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_IPU_READS] = {98, 1, 2, 0},
        [AMDF_METRICS_AVERAGE_IPU_WRITES] = {100, 1, 2, 0},
        [AMDF_METRICS_SYSTEM_CLOCK_COUNTER] = {104, 1, 8, 0},
        [AMDF_METRICS_TIME_FILTER_ALPHAVALUE] = {256, 1, 4, 0},
};

typedef struct amdf_gpu_kfd_metrics_layout_t {
  // Native format/content revision, encoded as format << 8 | content.
  uint16_t version;
  // Exact native record size, including alignment padding.
  uint16_t length;
  // Immutable diagnostic representation key.
  const char* name;
  // Indexed native field locations; shared by representation-compatible tails.
  const amdf_gpu_kfd_metrics_field_t* fields;
} amdf_gpu_kfd_metrics_layout_t;

static const amdf_gpu_kfd_metrics_layout_t amdf_gpu_kfd_metrics_layouts[] = {
    {0x0100, 80, "1.0", amdf_gpu_kfd_metrics_layout_1_0},
    {0x0101, 96, "1.1", amdf_gpu_kfd_metrics_layout_1_1},
    {0x0102, 104, "1.2", amdf_gpu_kfd_metrics_layout_1_2},
    {0x0103, 120, "1.3", amdf_gpu_kfd_metrics_layout_1_2},
    {0x0104, 288, "1.4", amdf_gpu_kfd_metrics_layout_1_4},
    {0x0105, 360, "1.5", amdf_gpu_kfd_metrics_layout_1_5},
    {0x0106, 1664, "1.6", amdf_gpu_kfd_metrics_layout_1_6},
    {0x0107, 2208, "1.7", amdf_gpu_kfd_metrics_layout_1_7},
    {0x0108, 3872, "1.8", amdf_gpu_kfd_metrics_layout_1_7},
    {0x0200, 120, "2.0", amdf_gpu_kfd_metrics_layout_2_0},
    {0x0201, 120, "2.1", amdf_gpu_kfd_metrics_layout_2_1},
    {0x0202, 128, "2.2", amdf_gpu_kfd_metrics_layout_2_1},
    {0x0203, 152, "2.3", amdf_gpu_kfd_metrics_layout_2_3},
    {0x0204, 168, "2.4", amdf_gpu_kfd_metrics_layout_2_3},
    {0x0300, 264, "3.0", amdf_gpu_kfd_metrics_layout_3_0},
};

// The native producer, not the binary layout alone, determines physical
// meaning. SMU15 reuses v3.0 clock members for targets and effective clocks
// across driver implementations; Aldebaran writes synthetic zero power on its
// secondary die.
enum amdf_gpu_kfd_metrics_profile_e {
  AMDF_METRICS_PROFILE_NONE,
  AMDF_METRICS_PROFILE_DISCRETE,
  AMDF_METRICS_PROFILE_ARCTURUS,
  AMDF_METRICS_PROFILE_ALDEBARAN,
  AMDF_METRICS_PROFILE_MULTI_DIE,
  AMDF_METRICS_PROFILE_SMU15_MULTI_DIE,
  AMDF_METRICS_PROFILE_CYAN_SKILLFISH,
  AMDF_METRICS_PROFILE_VANGOGH,
  AMDF_METRICS_PROFILE_RENOIR,
  AMDF_METRICS_PROFILE_YELLOW_CARP,
  AMDF_METRICS_PROFILE_SMU13_APU4,
  AMDF_METRICS_PROFILE_SMU13_APU5,
  AMDF_METRICS_PROFILE_SMU14_APU,
  AMDF_METRICS_PROFILE_SMU15_APU,
};

static uint32_t amdf_gpu_kfd_metrics_profile(uint32_t version) {
  switch (version) {
    case 0x0b0000:
    case 0x0b0005:
    case 0x0b0009:
    case 0x0b0007:
    case 0x0b000b:
    case 0x0b000c:
    case 0x0b000d:
    case 0x0d0000:
    case 0x0d0007:
    case 0x0d000a:
    case 0x0e0002:
    case 0x0e0003:
      return AMDF_METRICS_PROFILE_DISCRETE;
    case 0x0b0002:
      return AMDF_METRICS_PROFILE_ARCTURUS;
    case 0x0d0002:
      return AMDF_METRICS_PROFILE_ALDEBARAN;
    case 0x0d0006:
    case 0x0d000c:
    case 0x0d000e:
      return AMDF_METRICS_PROFILE_MULTI_DIE;
    case 0x0f0008:
      return AMDF_METRICS_PROFILE_SMU15_MULTI_DIE;
    case 0x0b0008:
      return AMDF_METRICS_PROFILE_CYAN_SKILLFISH;
    case 0x0b0500:
    case 0x0b0502:
      return AMDF_METRICS_PROFILE_VANGOGH;
    case 0x0c0000:
    case 0x0c0001:
      return AMDF_METRICS_PROFILE_RENOIR;
    case 0x0d0001:
    case 0x0d0003:
    case 0x0d0008:
      return AMDF_METRICS_PROFILE_YELLOW_CARP;
    case 0x0d0004:
    case 0x0d000b:
      return AMDF_METRICS_PROFILE_SMU13_APU4;
    case 0x0d0005:
      return AMDF_METRICS_PROFILE_SMU13_APU5;
    case 0x0e0000:
    case 0x0e0001:
    case 0x0e0004:
    case 0x0e0005:
      return AMDF_METRICS_PROFILE_SMU14_APU;
    case 0x0f0000:
    case 0x0f0005:
    case 0x0f0009:
      return AMDF_METRICS_PROFILE_SMU15_APU;
    default:
      return AMDF_METRICS_PROFILE_NONE;
  }
}

static bool amdf_gpu_kfd_metrics_is_multi_die(uint32_t profile) {
  return profile == AMDF_METRICS_PROFILE_MULTI_DIE ||
         profile == AMDF_METRICS_PROFILE_SMU15_MULTI_DIE;
}

bool amdf_gpu_kfd_metrics_supports_producer(uint32_t mp1_version) {
  return amdf_gpu_kfd_metrics_profile(mp1_version) != AMDF_METRICS_PROFILE_NONE;
}

static bool amdf_gpu_kfd_metrics_supports_version(uint32_t profile,
                                                  uint16_t version) {
  if (profile == AMDF_METRICS_PROFILE_NONE) {
    return false;
  }
  if (amdf_gpu_kfd_metrics_is_multi_die(profile)) {
    return version >= 0x0104 && version <= 0x0109;
  }
  if (profile <= AMDF_METRICS_PROFILE_ALDEBARAN) {
    return version >= 0x0100 && version <= 0x0103;
  }
  if (profile <= AMDF_METRICS_PROFILE_SMU13_APU5) {
    return version >= 0x0200 && version <= 0x0204;
  }
  return version == 0x0300;
}

#define FIELD_BIT(name) (UINT64_C(1) << AMDF_METRICS_##name)

static uint64_t amdf_gpu_kfd_metrics_fields(uint32_t profile) {
  const uint64_t driver_time = FIELD_BIT(SYSTEM_CLOCK_COUNTER);
  const uint64_t gpu_clocks = FIELD_BIT(AVERAGE_GFXCLK_FREQUENCY) |
                              FIELD_BIT(AVERAGE_SOCCLK_FREQUENCY) |
                              FIELD_BIT(AVERAGE_UCLK_FREQUENCY);
  const uint64_t current_clocks = FIELD_BIT(CURRENT_GFXCLK) |
                                  FIELD_BIT(CURRENT_SOCCLK) |
                                  FIELD_BIT(CURRENT_UCLK);
  const uint64_t core = FIELD_BIT(TEMPERATURE_CORE) |
                        FIELD_BIT(AVERAGE_CORE_POWER) |
                        FIELD_BIT(CURRENT_CORECLK);
  if (amdf_gpu_kfd_metrics_is_multi_die(profile)) {
    return driver_time | current_clocks | FIELD_BIT(TEMPERATURE_HOTSPOT) |
           FIELD_BIT(TEMPERATURE_MEM) | FIELD_BIT(AVERAGE_GFX_ACTIVITY) |
           FIELD_BIT(AVERAGE_UMC_ACTIVITY) | FIELD_BIT(CURR_SOCKET_POWER) |
           FIELD_BIT(ENERGY_ACCUMULATOR) | FIELD_BIT(FIRMWARE_TIMESTAMP) |
           (profile == AMDF_METRICS_PROFILE_SMU15_MULTI_DIE
                ? FIELD_BIT(TEMPERATURE_HBM)
                : 0);
  }
  if (profile <= AMDF_METRICS_PROFILE_ALDEBARAN) {
    uint64_t fields =
        driver_time | gpu_clocks | current_clocks |
        FIELD_BIT(TEMPERATURE_EDGE) | FIELD_BIT(TEMPERATURE_HOTSPOT) |
        FIELD_BIT(TEMPERATURE_MEM) | FIELD_BIT(AVERAGE_GFX_ACTIVITY) |
        FIELD_BIT(AVERAGE_UMC_ACTIVITY);
    if (profile == AMDF_METRICS_PROFILE_ALDEBARAN) {
      // The snapshot has no primary-die indication. Its zero power/energy
      // cannot distinguish unsupported secondary-die data from a measurement.
      fields |= FIELD_BIT(TEMPERATURE_HBM) | FIELD_BIT(FIRMWARE_TIMESTAMP);
    } else {
      fields |= FIELD_BIT(AVERAGE_SOCKET_POWER);
    }
    // Legacy energy widths and scales vary by firmware producer. Only the
    // exact 2^-16 J multi-die convention is admitted above.
    return fields;
  }
  if (profile <= AMDF_METRICS_PROFILE_SMU13_APU5) {
    uint64_t fields =
        driver_time | gpu_clocks | FIELD_BIT(TEMPERATURE_GFX) |
        FIELD_BIT(TEMPERATURE_SOC) | FIELD_BIT(AVERAGE_GFX_ACTIVITY) |
        FIELD_BIT(AVERAGE_SOCKET_POWER) | FIELD_BIT(AVERAGE_SOC_POWER) |
        FIELD_BIT(AVERAGE_GFX_POWER);
    if (profile != AMDF_METRICS_PROFILE_SMU13_APU5) {
      fields |= core;
    }
    if (profile == AMDF_METRICS_PROFILE_CYAN_SKILLFISH ||
        profile == AMDF_METRICS_PROFILE_VANGOGH) {
      fields |= current_clocks;
    }
    if (profile == AMDF_METRICS_PROFILE_VANGOGH) {
      fields |= FIELD_BIT(AVERAGE_CPU_POWER) |
                FIELD_BIT(AVERAGE_TEMPERATURE_GFX) |
                FIELD_BIT(AVERAGE_TEMPERATURE_SOC) |
                FIELD_BIT(AVERAGE_TEMPERATURE_CORE);
    }
    if (profile == AMDF_METRICS_PROFILE_RENOIR) {
      fields &=
          ~(FIELD_BIT(AVERAGE_GFX_POWER) | FIELD_BIT(AVERAGE_UCLK_FREQUENCY));
      fields |= current_clocks | FIELD_BIT(AVERAGE_CPU_POWER) |
                FIELD_BIT(AVERAGE_FCLK_FREQUENCY) | FIELD_BIT(CURRENT_FCLK);
    }
    // Other v2 producers duplicate MemclkFrequency into their fclk members.
    // Those aliases are not independent fabric-clock measurements.
    return fields;
  }
  uint64_t fields =
      driver_time | FIELD_BIT(TEMPERATURE_GFX) | FIELD_BIT(TEMPERATURE_SOC) |
      FIELD_BIT(TEMPERATURE_CORE) | FIELD_BIT(TEMPERATURE_SKIN) |
      FIELD_BIT(AVERAGE_GFX_ACTIVITY) | FIELD_BIT(AVERAGE_CORE_C0_ACTIVITY) |
      FIELD_BIT(AVERAGE_SOCKET_POWER) | FIELD_BIT(AVERAGE_APU_POWER) |
      FIELD_BIT(AVERAGE_GFX_POWER) | FIELD_BIT(AVERAGE_CORE_POWER) |
      FIELD_BIT(AVERAGE_SYS_POWER) | FIELD_BIT(AVERAGE_DRAM_READS) |
      FIELD_BIT(AVERAGE_DRAM_WRITES);
  if (profile == AMDF_METRICS_PROFILE_SMU14_APU) {
    // All clock members in this producer are targets. Core power is a native
    // CAC estimate; the aggregate and NPU powers have their own filter.
    fields |= FIELD_BIT(AVERAGE_IPU_ACTIVITY) | FIELD_BIT(AVERAGE_IPU_POWER) |
              FIELD_BIT(AVERAGE_ALL_CORE_POWER) | FIELD_BIT(AVERAGE_IPU_READS) |
              FIELD_BIT(AVERAGE_IPU_WRITES) | FIELD_BIT(TIME_FILTER_ALPHAVALUE);
  }
  // SMU15 reused v3.0 when replacing filtered targets with effective-clock
  // averages. Neither MP1 nor the table header distinguishes those driver
  // implementations, so clock fields have no portable measured meaning.
  // Its all-core power also became a partial sum over at most 16 active cores;
  // that cannot describe power of the entire CPU domain.
  return fields;
}

#undef FIELD_BIT

static uint32_t amdf_gpu_kfd_metrics_core_count(uint32_t profile) {
  switch (profile) {
    case AMDF_METRICS_PROFILE_CYAN_SKILLFISH:
      return 6;
    case AMDF_METRICS_PROFILE_VANGOGH:
      return 4;
    case AMDF_METRICS_PROFILE_RENOIR:
    case AMDF_METRICS_PROFILE_YELLOW_CARP:
    case AMDF_METRICS_PROFILE_SMU13_APU4:
      return 8;
    case AMDF_METRICS_PROFILE_SMU14_APU:
    case AMDF_METRICS_PROFILE_SMU15_APU:
      return 16;
    default:
      return 0;
  }
}

// Native records are little-endian on this x86-only provider. memcpy supports
// the unaligned inline u64 payloads in tagged snapshots without aliasing casts.
static uint64_t amdf_gpu_kfd_metrics_load(const uint8_t* data, uint32_t width) {
  uint64_t value = 0;
  memcpy(&value, data, width);
  return value;
}

static amdf_status_t amdf_gpu_kfd_metrics_malformed(void) {
  return amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EPROTO);
}

static const amdf_gpu_kfd_metrics_layout_t* amdf_gpu_kfd_metrics_layout(
    uint16_t version) {
  for (uint32_t i = 0; i < sizeof(amdf_gpu_kfd_metrics_layouts) /
                               sizeof(amdf_gpu_kfd_metrics_layouts[0]);
       ++i) {
    if (amdf_gpu_kfd_metrics_layouts[i].version == version) {
      return &amdf_gpu_kfd_metrics_layouts[i];
    }
  }
  return NULL;
}

static amdf_status_t amdf_gpu_kfd_metrics_decode_tagged(
    uint32_t length, const uint8_t* data, amdf_gpu_kfd_metrics_t* metrics) {
  if (length < 8) {
    return amdf_gpu_kfd_metrics_malformed();
  }
  const uint32_t count = (uint32_t)amdf_gpu_kfd_metrics_load(data + 4, 4);
  // Every attribute needs an eight-byte tag and at least one payload byte.
  // This also rejects negative native signed counts without a signed cast.
  if (count > (length - 8) / 9) {
    return amdf_gpu_kfd_metrics_malformed();
  }
  uint64_t seen[16] = {0};  // Native attribute IDs are ten bits.
  uint32_t position = 8;
  for (uint32_t i = 0; i < count; ++i) {
    if (length - position < 8) {
      return amdf_gpu_kfd_metrics_malformed();
    }
    const uint64_t tag = amdf_gpu_kfd_metrics_load(data + position, 8);
    position += 8;
    const uint32_t unit = (uint32_t)((tag >> 24) & 255);
    const uint32_t type = (uint32_t)((tag >> 20) & 15);
    const uint32_t id = (uint32_t)((tag >> 10) & 1023);
    const uint32_t instances = (uint32_t)(tag & 1023);
    if ((tag >> 32) != 0 || type > 7 || instances == 0 ||
        (seen[id / 64] & (UINT64_C(1) << (id % 64))) != 0) {
      return amdf_gpu_kfd_metrics_malformed();
    }
    seen[id / 64] |= UINT64_C(1) << (id % 64);
    const uint32_t width = 1u << (type / 2);
    const uint32_t bytes = width * instances;
    if (bytes > length - position) {
      return amdf_gpu_kfd_metrics_malformed();
    }
    for (uint32_t field = 0; field < AMDF_METRICS_FIELD_COUNT; ++field) {
      const amdf_gpu_kfd_metric_definition_t* definition =
          &amdf_gpu_kfd_metric_definitions[field];
      if (definition->tag_id == UINT16_MAX || definition->tag_id != id) {
        continue;
      }
      if (type != definition->tag_type || unit != definition->tag_unit) {
        return amdf_gpu_kfd_metrics_malformed();
      }
      const bool is_array =
          field == AMDF_METRICS_CURRENT_GFXCLK ||
          field == AMDF_METRICS_CURRENT_SOCCLK ||
          field == AMDF_METRICS_TEMPERATURE_HBM ||
          (field == AMDF_METRICS_CURRENT_UCLK &&
           metrics->profile == AMDF_METRICS_PROFILE_SMU15_MULTI_DIE);
      if (!is_array && instances != 1) {
        return amdf_gpu_kfd_metrics_malformed();
      }
      metrics->fields[field] = (amdf_gpu_kfd_metrics_field_t){
          .offset = (uint16_t)position,
          .count = (uint16_t)instances,
          .width = (uint8_t)width,
          .is_array = is_array,
      };
      break;
    }
    position += bytes;
  }
  return position == length ? AMDF_STATUS_OK : amdf_gpu_kfd_metrics_malformed();
}

amdf_status_t amdf_gpu_kfd_metrics_decode(uint32_t mp1_version, uint32_t length,
                                          const uint8_t* data,
                                          amdf_gpu_kfd_metrics_t* out_metrics) {
  const uint32_t profile = amdf_gpu_kfd_metrics_profile(mp1_version);
  if (profile == AMDF_METRICS_PROFILE_NONE || length == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (length < 4 || length >= AMDF_GPU_KFD_METRICS_CAPACITY ||
      amdf_gpu_kfd_metrics_load(data, 2) != length) {
    return amdf_gpu_kfd_metrics_malformed();
  }
  const uint16_t version = (uint16_t)((uint16_t)data[2] << 8 | data[3]);
  if (!amdf_gpu_kfd_metrics_supports_version(profile, version)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_gpu_kfd_metrics_t metrics = {
      .data = data,
      .source_id = AMDF_GPU_KFD_METRICS_SOURCE_BASE |
                   (uint64_t)mp1_version << 16 | version,
      .profile = profile,
      .version = version,
  };
  if (version == 0x0109) {
    const amdf_status_t status =
        amdf_gpu_kfd_metrics_decode_tagged(length, data, &metrics);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    metrics.schema = "1.9";
  } else {
    const amdf_gpu_kfd_metrics_layout_t* layout =
        amdf_gpu_kfd_metrics_layout(version);
    if (layout->length != length) {
      return amdf_gpu_kfd_metrics_malformed();
    }
    metrics.schema = layout->name;
    memcpy(metrics.fields, layout->fields, sizeof(metrics.fields));
  }
  const uint64_t supported_fields = amdf_gpu_kfd_metrics_fields(profile);
  for (uint32_t i = 0; i < AMDF_METRICS_FIELD_COUNT; ++i) {
    if ((supported_fields & (UINT64_C(1) << i)) == 0) {
      metrics.fields[i] = (amdf_gpu_kfd_metrics_field_t){0};
    } else if (amdf_gpu_kfd_metric_definitions[i].domain ==
               AMDF_TELEMETRY_DOMAIN_CPU_CORE) {
      metrics.fields[i].count =
          metrics.fields[i].count == 0
              ? 0
              : (uint16_t)amdf_gpu_kfd_metrics_core_count(profile);
    }
  }
  *out_metrics = metrics;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_kfd_metrics_validate_selection(
    amdf_telemetry_source_id_t source_id, uint32_t count,
    const amdf_telemetry_metric_id_t* metric_ids) {
  const uint32_t mp1 = (uint32_t)((source_id >> 16) & 0xffffff);
  const uint32_t profile = amdf_gpu_kfd_metrics_profile(mp1);
  const uint16_t version = (uint16_t)source_id;
  if ((source_id & UINT64_C(0xffffff0000000000)) !=
          AMDF_GPU_KFD_METRICS_SOURCE_BASE ||
      !amdf_gpu_kfd_metrics_supports_version(profile, version)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }
  const uint64_t supported_fields = amdf_gpu_kfd_metrics_fields(profile);
  const amdf_gpu_kfd_metrics_layout_t* layout =
      version == 0x0109 ? NULL : amdf_gpu_kfd_metrics_layout(version);
  for (uint32_t i = 0; i < count; ++i) {
    const uint64_t field = metric_ids[i] >> 16;
    const uint32_t element = (uint32_t)(metric_ids[i] & 0xffff);
    if (field >= AMDF_METRICS_FIELD_COUNT || element == 0 ||
        amdf_gpu_kfd_metric_definitions[field].kind == 0 ||
        (supported_fields & (UINT64_C(1) << field)) == 0 ||
        (layout != NULL && element > layout->fields[field].count) ||
        (layout == NULL &&
         (amdf_gpu_kfd_metric_definitions[field].tag_id == UINT16_MAX ||
          element > 1023))) {
      return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
    }
    if (amdf_gpu_kfd_metric_definitions[field].domain ==
            AMDF_TELEMETRY_DOMAIN_CPU_CORE &&
        element > amdf_gpu_kfd_metrics_core_count(profile)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
    }
  }
  return AMDF_STATUS_OK;
}

static bool amdf_gpu_kfd_metrics_is_filtered(
    const amdf_gpu_kfd_metrics_t* metrics, uint32_t field) {
  const uint32_t kind = amdf_gpu_kfd_metric_definitions[field].kind;
  return metrics->profile == AMDF_METRICS_PROFILE_SMU14_APU &&
         ((kind == AMDF_TELEMETRY_METRIC_POWER &&
           field != AMDF_METRICS_AVERAGE_CORE_POWER) ||
          kind == AMDF_TELEMETRY_METRIC_ACTIVITY_COMPUTE ||
          kind == AMDF_TELEMETRY_METRIC_ACTIVITY_ACTIVE ||
          kind == AMDF_TELEMETRY_METRIC_TRAFFIC_READ ||
          kind == AMDF_TELEMETRY_METRIC_TRAFFIC_WRITE);
}

static amdf_telemetry_metric_info_t amdf_gpu_kfd_metrics_describe_field(
    const amdf_gpu_kfd_metrics_t* metrics, uint32_t field, uint32_t element) {
  const amdf_gpu_kfd_metric_definition_t* definition =
      &amdf_gpu_kfd_metric_definitions[field];
  const bool apu = metrics->version >= 0x0200;
  const bool multi_die = amdf_gpu_kfd_metrics_is_multi_die(metrics->profile);
  amdf_telemetry_metric_info_t result = {
      .source_id = metrics->source_id,
      .source_name = "gpu_metrics",
      .source_flags = AMDF_TELEMETRY_SOURCE_FLAG_NATIVE_MAY_BLOCK |
                      AMDF_TELEMETRY_SOURCE_FLAG_MAY_WAKE_DEVICE |
                      AMDF_TELEMETRY_SOURCE_FLAG_NATIVE_SNAPSHOT,
      .domain = definition->domain,
      .metric_id = (uint64_t)field << 16 | (element + 1),
      .domain_instance = metrics->fields[field].is_array ? element : 0,
      .instance_scope = AMDF_TELEMETRY_INSTANCE_SCOPE_DEVICE,
      .kind = definition->kind,
      .format = {.kind = AMDF_OBSERVATION_VALUE_KIND_U64,
                 .scale_numerator = 1,
                 .scale_denominator = 1},
      .name = definition->name,
      .origin = {.interface_name = "linux.amdgpu.gpu_metrics",
                 .schema = metrics->schema,
                 .field = definition->name,
                 .field_index =
                     metrics->fields[field].is_array ? element : UINT32_MAX},
  };
  switch (result.kind) {
    case AMDF_TELEMETRY_METRIC_CLOCK_COMPUTE:
    case AMDF_TELEMETRY_METRIC_CLOCK_MEMORY:
    case AMDF_TELEMETRY_METRIC_CLOCK_MEMORY_CONTROLLER:
    case AMDF_TELEMETRY_METRIC_CLOCK_SOC:
    case AMDF_TELEMETRY_METRIC_CLOCK_FABRIC:
      result.format.unit = AMDF_OBSERVATION_UNIT_HERTZ;
      result.format.scale_numerator = 1000000;
      break;
    case AMDF_TELEMETRY_METRIC_TEMPERATURE_EDGE:
    case AMDF_TELEMETRY_METRIC_TEMPERATURE_HOTSPOT:
    case AMDF_TELEMETRY_METRIC_TEMPERATURE_MEMORY:
    case AMDF_TELEMETRY_METRIC_TEMPERATURE:
      result.format.unit = AMDF_OBSERVATION_UNIT_CELSIUS;
      result.format.scale_denominator = apu ? 100 : 1;
      break;
    case AMDF_TELEMETRY_METRIC_POWER:
      result.format.unit = AMDF_OBSERVATION_UNIT_WATTS;
      result.format.scale_denominator = apu ? 1000 : 1;
      break;
    case AMDF_TELEMETRY_METRIC_ENERGY:
      result.format.unit = AMDF_OBSERVATION_UNIT_JOULES;
      result.format.scale_denominator = 65536;
      result.format.semantic = AMDF_OBSERVATION_VALUE_SEMANTIC_CUMULATIVE;
      result.format.counter_bit_count = 64;
      break;
    case AMDF_TELEMETRY_METRIC_ACTIVITY_COMPUTE:
    case AMDF_TELEMETRY_METRIC_ACTIVITY_MEMORY:
    case AMDF_TELEMETRY_METRIC_ACTIVITY_ACTIVE:
      result.format.unit = AMDF_OBSERVATION_UNIT_RATIO;
      result.format.scale_denominator = 100;
      break;
    case AMDF_TELEMETRY_METRIC_TRAFFIC_READ:
    case AMDF_TELEMETRY_METRIC_TRAFFIC_WRITE:
      result.format.unit = AMDF_OBSERVATION_UNIT_BYTES_PER_SECOND;
      result.format.scale_numerator = 1000000;
      break;
  }
  if (apu && field == AMDF_METRICS_AVERAGE_SOCKET_POWER) {
    result.domain = AMDF_TELEMETRY_DOMAIN_POWER_BUDGET;
  }
  if (multi_die && field == AMDF_METRICS_TEMPERATURE_HOTSPOT) {
    result.domain = AMDF_TELEMETRY_DOMAIN_PACKAGE;
  }
  if (result.domain == AMDF_TELEMETRY_DOMAIN_CPU ||
      (multi_die && (field == AMDF_METRICS_AVERAGE_GFX_ACTIVITY ||
                     field == AMDF_METRICS_AVERAGE_UMC_ACTIVITY ||
                     field == AMDF_METRICS_TEMPERATURE_MEM)) ||
      field == AMDF_METRICS_AVERAGE_DRAM_READS ||
      field == AMDF_METRICS_AVERAGE_DRAM_WRITES) {
    result.domain_instance = AMDF_TELEMETRY_ALL_INSTANCES;
  }
  if (metrics->profile == AMDF_METRICS_PROFILE_RENOIR &&
      field == AMDF_METRICS_CURRENT_UCLK) {
    result.kind = AMDF_TELEMETRY_METRIC_CLOCK_MEMORY_CONTROLLER;
    result.domain = AMDF_TELEMETRY_DOMAIN_MEMORY_CONTROLLER;
  }
  if (metrics->profile == AMDF_METRICS_PROFILE_SMU15_APU) {
    // The same native revision covers filtered and accumulator-based driver
    // producers. Retain their shared physical units without claiming which
    // integration rule produced this observation.
    if (result.domain == AMDF_TELEMETRY_DOMAIN_CPU_CORE) {
      result.instance_scope = AMDF_TELEMETRY_INSTANCE_SCOPE_SNAPSHOT;
    }
  } else if (amdf_gpu_kfd_metrics_is_filtered(metrics, field)) {
    result.format.semantic = AMDF_OBSERVATION_VALUE_SEMANTIC_FILTERED;
  } else if (field == AMDF_METRICS_AVERAGE_TEMPERATURE_GFX ||
             field == AMDF_METRICS_AVERAGE_TEMPERATURE_SOC ||
             field == AMDF_METRICS_AVERAGE_TEMPERATURE_CORE) {
    result.format.semantic = AMDF_OBSERVATION_VALUE_SEMANTIC_AVERAGE;
  }
  return result;
}

uint32_t amdf_gpu_kfd_metrics_describe(
    const amdf_gpu_kfd_metrics_t* metrics, uint32_t capacity,
    amdf_telemetry_metric_info_t* out_metrics) {
  uint32_t count = 0;
  for (uint32_t field = 0; field < AMDF_METRICS_FIELD_COUNT; ++field) {
    if (amdf_gpu_kfd_metric_definitions[field].kind == 0) {
      continue;
    }
    for (uint32_t element = 0; element < metrics->fields[field].count;
         ++element) {
      if (count < capacity) {
        out_metrics[count] =
            amdf_gpu_kfd_metrics_describe_field(metrics, field, element);
      }
      ++count;
    }
  }
  return count;
}

static uint64_t amdf_gpu_kfd_metrics_raw(const amdf_gpu_kfd_metrics_t* metrics,
                                         uint32_t field, uint32_t element) {
  const amdf_gpu_kfd_metrics_field_t* location = &metrics->fields[field];
  return amdf_gpu_kfd_metrics_load(
      metrics->data + location->offset + element * location->width,
      location->width);
}

static bool amdf_gpu_kfd_metrics_unavailable(
    const amdf_gpu_kfd_metrics_t* metrics, uint32_t field, uint64_t value) {
  // AMD SMI's v3.0 contract identifies zero core/skin temperature as
  // unmeasured, in addition to the table initializer's all-ones sentinel.
  // Other fields (including power and activity) retain valid zero readings.
  const bool unmeasured_temperature = metrics->version == 0x0300 &&
                                      value == 0 &&
                                      (field == AMDF_METRICS_TEMPERATURE_CORE ||
                                       field == AMDF_METRICS_TEMPERATURE_SKIN);
  return metrics->fields[field].count == 0 ||
         value == (UINT64_MAX >> (64 - metrics->fields[field].width * 8)) ||
         unmeasured_temperature;
}

static amdf_telemetry_timestamp_t amdf_gpu_kfd_metrics_timestamp(
    const amdf_gpu_kfd_metrics_t* metrics, uint32_t field) {
  amdf_telemetry_timestamp_t result = {0};
  if (metrics->fields[field].count == 0) {
    return result;
  }
  const uint64_t value = amdf_gpu_kfd_metrics_raw(metrics, field, 0);
  if (amdf_gpu_kfd_metrics_unavailable(metrics, field, value)) {
    return result;
  }
  const bool driver = field == AMDF_METRICS_SYSTEM_CLOCK_COUNTER;
  result.domain =
      driver ? AMDF_TELEMETRY_CLOCK_SYSTEM : AMDF_TELEMETRY_CLOCK_SOURCE;
  result.point = driver ? AMDF_TELEMETRY_TIMESTAMP_POINT_DRIVER_READ
                        : AMDF_TELEMETRY_TIMESTAMP_POINT_UNSPECIFIED;
  result.clock.frequency_hz = driver ? 1000000000 : 100000000;
  result.clock.counter_bit_count = 64;
  result.value = value;
  return result;
}

amdf_status_t amdf_gpu_kfd_metrics_read(
    const amdf_gpu_kfd_metrics_t* metrics, uint32_t count,
    const amdf_telemetry_metric_id_t* metric_ids,
    amdf_telemetry_value_t* values, amdf_telemetry_sample_t* sample) {
  // This is a caller selection boundary. The byte parser already established
  // all field locations, so successful selection leaves only infallible loads.
  for (uint32_t i = 0; i < count; ++i) {
    const uint64_t field = metric_ids[i] >> 16;
    const uint32_t element = (uint32_t)(metric_ids[i] & 0xffff);
    if (field >= AMDF_METRICS_FIELD_COUNT || element == 0 ||
        amdf_gpu_kfd_metric_definitions[field].kind == 0 ||
        element > metrics->fields[field].count) {
      return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
    }
  }
  uint64_t filter_ns = 0;
  bool has_filter = false;
  if (metrics->fields[AMDF_METRICS_TIME_FILTER_ALPHAVALUE].count != 0) {
    const uint64_t alpha = amdf_gpu_kfd_metrics_raw(
        metrics, AMDF_METRICS_TIME_FILTER_ALPHAVALUE, 0);
    if (!amdf_gpu_kfd_metrics_unavailable(
            metrics, AMDF_METRICS_TIME_FILTER_ALPHAVALUE, alpha)) {
      filter_ns = alpha * 1000;
      has_filter = true;
    }
  }
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t field = (uint32_t)(metric_ids[i] >> 16);
    const uint32_t element = (uint32_t)(metric_ids[i] & 0xffff) - 1;
    const uint64_t raw = amdf_gpu_kfd_metrics_raw(metrics, field, element);
    const bool unavailable =
        amdf_gpu_kfd_metrics_unavailable(metrics, field, raw);
    amdf_telemetry_value_t value = {
        .status = unavailable ? amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND)
                              : AMDF_STATUS_OK,
        .value = {.u64 = unavailable ? 0 : raw},
    };
    if (!unavailable && has_filter &&
        amdf_gpu_kfd_metrics_is_filtered(metrics, field)) {
      value.duration_kind = AMDF_TELEMETRY_DURATION_FILTER_TIME_CONSTANT;
      value.duration_ns = filter_ns;
    }
    values[i] = value;
  }
  sample->driver_time = amdf_gpu_kfd_metrics_timestamp(
      metrics, AMDF_METRICS_SYSTEM_CLOCK_COUNTER);
  sample->source_time =
      amdf_gpu_kfd_metrics_timestamp(metrics, AMDF_METRICS_FIRMWARE_TIMESTAMP);
  return AMDF_STATUS_OK;
}
