// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/gpu/umd/telemetry.h"

#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>

#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/observation.h"

// Each sysfs attribute is a separate source. The source identity names its
// interpretation, independent of the kernel's ephemeral hwmon directory number.
#define AMDF_GPU_KFD_HWMON_SOURCE_BASE UINT64_C(0x0100000000000000)

// A hwmon attribute names a native sensor route, not a portable meaning. These
// profiles follow amdgpu_smu.c's MP1 producer selection and each producer's
// read_sensor/get_smu_metrics_data mapping and firmware interface declarations.
typedef uint32_t amdf_gpu_kfd_hwmon_profile_t;
enum amdf_gpu_kfd_hwmon_profile_e {
  AMDF_GPU_KFD_HWMON_PROFILE_NONE = 0,
  AMDF_GPU_KFD_HWMON_PROFILE_DISCRETE = 1,
  AMDF_GPU_KFD_HWMON_PROFILE_APU = 2,
  AMDF_GPU_KFD_HWMON_PROFILE_RENOIR = 3,
  AMDF_GPU_KFD_HWMON_PROFILE_SMU14_APU = 4,
  AMDF_GPU_KFD_HWMON_PROFILE_MULTI_DIE = 5,
  AMDF_GPU_KFD_HWMON_PROFILE_MULTI_DIE_NO_GFXCLK = 6,
};

static amdf_gpu_kfd_hwmon_profile_t amdf_gpu_kfd_hwmon_profile(
    uint32_t version) {
  switch (version) {
    // Navi, Sienna Cichlid, Arcturus, Aldebaran, and discrete SMU13/14.
    case 0x0b0000:
    case 0x0b0002:
    case 0x0b0005:
    case 0x0b0007:
    case 0x0b0009:
    case 0x0b000b:
    case 0x0b000c:
    case 0x0b000d:
    case 0x0d0000:
    case 0x0d0002:
    case 0x0d0007:
    case 0x0d000a:
    case 0x0e0002:
    case 0x0e0003:
      return AMDF_GPU_KFD_HWMON_PROFILE_DISCRETE;
    case 0x0d0006:
    case 0x0d000c:
    case 0x0d000e:
    case 0x0f0008:
      return AMDF_GPU_KFD_HWMON_PROFILE_MULTI_DIE;
    // Cyan Skillfish, Vangogh, Yellow Carp, and the SMU13 APU producers.
    case 0x0b0008:
    case 0x0b0500:
    case 0x0b0502:
    case 0x0d0001:
    case 0x0d0003:
    case 0x0d0004:
    case 0x0d0005:
    case 0x0d0008:
    case 0x0d000b:
      return AMDF_GPU_KFD_HWMON_PROFILE_APU;
    case 0x0c0000:
    case 0x0c0001:
      return AMDF_GPU_KFD_HWMON_PROFILE_RENOIR;
    case 0x0e0000:
    case 0x0e0001:
    case 0x0e0004:
    case 0x0e0005:
      return AMDF_GPU_KFD_HWMON_PROFILE_SMU14_APU;
    default:
      return AMDF_GPU_KFD_HWMON_PROFILE_NONE;
  }
}

static uint32_t amdf_gpu_kfd_hwmon_fields_for_profile(
    amdf_gpu_kfd_hwmon_profile_t profile) {
  switch (profile) {
    case AMDF_GPU_KFD_HWMON_PROFILE_DISCRETE:
      return 0x7f;
    case AMDF_GPU_KFD_HWMON_PROFILE_APU:
    case AMDF_GPU_KFD_HWMON_PROFILE_RENOIR:
      return 0x3f;
    case AMDF_GPU_KFD_HWMON_PROFILE_SMU14_APU:
      // GfxclkFrequency and MemclkFrequency are filtered target frequencies.
      // The measured-clock API cannot advertise either as an actual rate.
      return 0x3c;
    case AMDF_GPU_KFD_HWMON_PROFILE_MULTI_DIE:
      return 0x6b;
    case AMDF_GPU_KFD_HWMON_PROFILE_MULTI_DIE_NO_GFXCLK:
      // Older SMU13 firmware makes the driver return a synthetic zero clock.
      return 0x6a;
    default:
      return 0;
  }
}

typedef struct amdf_gpu_kfd_hwmon_field_t {
  // Stable native attribute name, also used as diagnostic provenance.
  const char* name;
  // Portable measurement meaning.
  amdf_telemetry_metric_kind_t kind;
  // Physical measurement domain.
  amdf_telemetry_domain_kind_t domain;
  // Signed temperature or unsigned clock/power representation.
  amdf_observation_value_kind_t value_kind;
  // Physical unit after dividing by denominator.
  amdf_observation_unit_t unit;
  // Exact native unit conversion denominator.
  uint32_t denominator;
} amdf_gpu_kfd_hwmon_field_t;

// amdgpu_pm.c defines these fixed channels and their units. This is the
// discrete-GPU interpretation; APU profiles refine the physical meanings below.
// The average/input routes can reach the same native quantity, so their
// spelling alone does not establish an aggregation or filter window.
static const amdf_gpu_kfd_hwmon_field_t amdf_gpu_kfd_hwmon_fields[] = {
    {"freq1_input", AMDF_TELEMETRY_METRIC_CLOCK_COMPUTE,
     AMDF_TELEMETRY_DOMAIN_GPU, AMDF_OBSERVATION_VALUE_KIND_U64,
     AMDF_OBSERVATION_UNIT_HERTZ, 1},
    {"freq2_input", AMDF_TELEMETRY_METRIC_CLOCK_MEMORY,
     AMDF_TELEMETRY_DOMAIN_MEMORY, AMDF_OBSERVATION_VALUE_KIND_U64,
     AMDF_OBSERVATION_UNIT_HERTZ, 1},
    {"power1_average", AMDF_TELEMETRY_METRIC_POWER,
     AMDF_TELEMETRY_DOMAIN_PACKAGE, AMDF_OBSERVATION_VALUE_KIND_U64,
     AMDF_OBSERVATION_UNIT_WATTS, 1000000},
    {"power1_input", AMDF_TELEMETRY_METRIC_POWER, AMDF_TELEMETRY_DOMAIN_PACKAGE,
     AMDF_OBSERVATION_VALUE_KIND_U64, AMDF_OBSERVATION_UNIT_WATTS, 1000000},
    {"temp1_input", AMDF_TELEMETRY_METRIC_TEMPERATURE_EDGE,
     AMDF_TELEMETRY_DOMAIN_GPU, AMDF_OBSERVATION_VALUE_KIND_I64,
     AMDF_OBSERVATION_UNIT_CELSIUS, 1000},
    {"temp2_input", AMDF_TELEMETRY_METRIC_TEMPERATURE_HOTSPOT,
     AMDF_TELEMETRY_DOMAIN_GPU, AMDF_OBSERVATION_VALUE_KIND_I64,
     AMDF_OBSERVATION_UNIT_CELSIUS, 1000},
    {"temp3_input", AMDF_TELEMETRY_METRIC_TEMPERATURE_MEMORY,
     AMDF_TELEMETRY_DOMAIN_MEMORY, AMDF_OBSERVATION_VALUE_KIND_I64,
     AMDF_OBSERVATION_UNIT_CELSIUS, 1000},
};

#define AMDF_GPU_KFD_HWMON_FIELD_COUNT \
  (sizeof(amdf_gpu_kfd_hwmon_fields) / sizeof(amdf_gpu_kfd_hwmon_fields[0]))

// One kernel getdents64 record. Reading into caller storage avoids the heap
// buffer owned by fdopendir; this cold scan retains no directory or sensor.
typedef struct amdf_gpu_kfd_directory_entry_t {
  // Native inode number.
  uint64_t inode;
  // Native directory cursor after this record.
  int64_t offset;
  // Complete padded record byte length.
  uint16_t record_length;
  // Native directory entry type.
  uint8_t type;
  // NUL-terminated name within record_length.
  char name[];
} amdf_gpu_kfd_directory_entry_t;

static amdf_status_t amdf_gpu_kfd_hwmon_read_text(int directory,
                                                  const char* name,
                                                  size_t capacity, char* text) {
  int descriptor = openat(directory, name, O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return amdf_linux_error(errno);
  }
  // A sysfs attribute is a single bounded record. EINTR remains an explicit
  // native failure; a sensor query never retries indefinitely in the library.
  const ssize_t length = read(descriptor, text, capacity - 1);
  amdf_status_t status = length < 0 ? amdf_linux_error(errno) : AMDF_STATUS_OK;
  if (length >= 0) {
    if (memchr(text, 0, (size_t)length) != NULL) {
      status = amdf_linux_error(EPROTO);
    }
    text[length] = 0;
    if ((size_t)length == capacity - 1) {
      status = amdf_linux_error(EOVERFLOW);
    }
  }
  const amdf_status_t close_status = amdf_linux_file_close(&descriptor);
  return amdf_status_is_ok(status) ? close_status : status;
}

static amdf_status_t amdf_gpu_kfd_hwmon_read_integer(
    int directory, const char* name, amdf_observation_value_kind_t kind,
    amdf_observation_value_t* out_value) {
  char text[64];
  const amdf_status_t status =
      amdf_gpu_kfd_hwmon_read_text(directory, name, sizeof(text), text);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  char* end = NULL;
  const char* digits = text;
  if (kind == AMDF_OBSERVATION_VALUE_KIND_I64 && *digits == '-') {
    ++digits;
  }
  if (*digits < '0' || *digits > '9') {
    return amdf_linux_error(EPROTO);
  }
  errno = 0;
  amdf_observation_value_t value = {0};
  if (kind == AMDF_OBSERVATION_VALUE_KIND_I64) {
    value.i64 = strtoll(text, &end, 10);
  } else {
    value.u64 = strtoull(text, &end, 10);
  }
  if (errno != 0 || end == text || (*end != 0 && *end != '\n') ||
      (*end == '\n' && end[1] != 0)) {
    return amdf_linux_error(EPROTO);
  }
  *out_value = value;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_kfd_hwmon_query_profile(
    int device_directory, amdf_gpu_kfd_hwmon_profile_t* out_profile) {
  // MP1_HWID is 1. Numeric paths also work before named IP symlinks existed.
  // This explicit telemetry query neither extends discovery nor retains state.
  int directory = openat(device_directory, "ip_discovery/die/0/1/0",
                         O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (directory < 0) {
    if (errno != ENOENT) {
      return amdf_linux_error(errno);
    }
    *out_profile = AMDF_GPU_KFD_HWMON_PROFILE_NONE;
    return AMDF_STATUS_OK;
  }
  const char* names[] = {"major", "minor", "revision"};
  uint32_t version = 0;
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0; amdf_status_is_ok(status) && i < 3; ++i) {
    amdf_observation_value_t value;
    status = amdf_gpu_kfd_hwmon_read_integer(
        directory, names[i], AMDF_OBSERVATION_VALUE_KIND_U64, &value);
    if (amdf_status_is_ok(status)) {
      if (value.u64 > UINT8_MAX) {
        status = amdf_linux_error(EPROTO);
      } else {
        version = (version << 8) | (uint32_t)value.u64;
      }
    }
  }
  const amdf_status_t close_status = amdf_linux_file_close(&directory);
  if (amdf_status_is_ok(status)) {
    status = close_status;
  }
  if (amdf_status_is_ok(status)) {
    amdf_gpu_kfd_hwmon_profile_t profile = amdf_gpu_kfd_hwmon_profile(version);
    if (version == 0x0d0006 || version == 0x0d000c) {
      // smu_v13_0_6_init_caps/smu_v13_0_12_init_caps suppress DPM below these
      // firmware revisions. Their GFXCLK sensor returns zero instead of an
      // unsupported error. The native firmware attribute disambiguates it.
      char text[32];
      status = amdf_gpu_kfd_hwmon_read_text(
          device_directory, "fw_version/smc_fw_version", sizeof(text), text);
      if (status == amdf_linux_error(ENOENT)) {
        profile = AMDF_GPU_KFD_HWMON_PROFILE_MULTI_DIE_NO_GFXCLK;
        status = AMDF_STATUS_OK;
      } else if (amdf_status_is_ok(status)) {
        char* end = NULL;
        errno = 0;
        const unsigned long long firmware = strtoull(text, &end, 16);
        if (strlen(text) != 11 || text[0] != '0' || text[1] != 'x' ||
            end != text + 10 || *end != '\n' || errno != 0 ||
            firmware > UINT32_MAX) {
          status = amdf_linux_error(EPROTO);
        } else if (firmware <
                   (version == 0x0d0006 ? 0x00552f00u : 0x00561900u)) {
          profile = AMDF_GPU_KFD_HWMON_PROFILE_MULTI_DIE_NO_GFXCLK;
        }
      }
    }
    if (amdf_status_is_ok(status)) {
      *out_profile = profile;
    }
  }
  return status;
}

static amdf_status_t amdf_gpu_kfd_hwmon_open(
    const amdf_platform_endpoint_t* endpoint, int* out_directory,
    amdf_gpu_kfd_hwmon_profile_t* out_profile) {
  // The live device's render file pins this DRM minor allocation. Unregister
  // removes its sysfs path; the minor cannot name a replacement while borrowed.
  char path[80];
  snprintf(path, sizeof(path), "dev/char/%u:%u/device",
           endpoint->info.native_identity.value.linux_device.major,
           endpoint->info.native_identity.value.linux_device.minor);
  int device_directory = openat(endpoint->instance->sysfs_descriptor, path,
                                O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (device_directory < 0) {
    return amdf_linux_error(errno);
  }
  amdf_gpu_kfd_hwmon_profile_t profile = AMDF_GPU_KFD_HWMON_PROFILE_NONE;
  amdf_status_t status =
      amdf_gpu_kfd_hwmon_query_profile(device_directory, &profile);
  int parent = -1;
  if (amdf_status_is_ok(status) && profile != AMDF_GPU_KFD_HWMON_PROFILE_NONE) {
    parent =
        openat(device_directory, "hwmon", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (parent < 0 && errno != ENOENT) {
      status = amdf_linux_error(errno);
    }
  }
  const amdf_status_t device_close = amdf_linux_file_close(&device_directory);
  if (amdf_status_is_ok(status)) {
    status = device_close;
  }
  if (amdf_status_is_ok(status) && parent < 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  int sensor_directory = -1;
  bool complete = false;
  while (amdf_status_is_ok(status) && !complete) {
    _Alignas(uint64_t) uint8_t entries[1024];
    const long length =
        syscall(SYS_getdents64, parent, entries, sizeof(entries));
    if (length < 0) {
      status = amdf_linux_error(errno);
    } else if (length == 0) {
      complete = true;
    }
    size_t position = 0;
    while (amdf_status_is_ok(status) && position < (size_t)length) {
      const amdf_gpu_kfd_directory_entry_t* entry =
          (const amdf_gpu_kfd_directory_entry_t*)(entries + position);
      const size_t remaining = (size_t)length - position;
      const size_t prefix = offsetof(amdf_gpu_kfd_directory_entry_t, name);
      if (remaining <= prefix || entry->record_length <= prefix ||
          entry->record_length > remaining || entry->record_length % 8 != 0 ||
          memchr(entry->name, 0, entry->record_length - prefix) == NULL) {
        status = amdf_linux_error(EPROTO);
        break;
      }
      position += entry->record_length;
      if (strncmp(entry->name, "hwmon", 5) != 0) {
        continue;
      }
      int candidate =
          openat(parent, entry->name, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
      if (candidate < 0) {
        status = amdf_linux_error(errno);
      } else {
        char name[32];
        status =
            amdf_gpu_kfd_hwmon_read_text(candidate, "name", sizeof(name), name);
        if (amdf_status_is_ok(status) && strcmp(name, "amdgpu\n") == 0) {
          if (sensor_directory >= 0) {
            // One amdgpu device owns one hwmon registration. Selecting one of
            // multiple registrations would leave physical attribution unknown.
            status = amdf_linux_error(EPROTO);
          } else {
            sensor_directory = candidate;
            candidate = -1;
          }
        }
        const amdf_status_t candidate_close = amdf_linux_file_close(&candidate);
        if (amdf_status_is_ok(status)) {
          status = candidate_close;
        }
      }
    }
  }
  const amdf_status_t parent_close = amdf_linux_file_close(&parent);
  if (amdf_status_is_ok(status)) {
    status = parent_close;
  }
  if (amdf_status_is_ok(status) && sensor_directory < 0) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (amdf_status_is_ok(status)) {
    *out_directory = sensor_directory;
    *out_profile = profile;
  } else {
    // Preserve the originating error; close consumes this unexposed descriptor.
    const amdf_status_t close_status = amdf_linux_file_close(&sensor_directory);
    (void)close_status;
  }
  return status;
}

static amdf_telemetry_metric_info_t amdf_gpu_kfd_hwmon_describe(
    amdf_gpu_kfd_hwmon_profile_t profile, uint32_t ordinal) {
  const amdf_gpu_kfd_hwmon_field_t* field = &amdf_gpu_kfd_hwmon_fields[ordinal];
  const amdf_telemetry_source_id_t id =
      AMDF_GPU_KFD_HWMON_SOURCE_BASE + (profile << 8) + ordinal + 1;
  amdf_telemetry_metric_info_t info = {
      .source_id = id,
      .source_name = field->name,
      .source_flags = AMDF_TELEMETRY_SOURCE_FLAG_NATIVE_MAY_BLOCK |
                      AMDF_TELEMETRY_SOURCE_FLAG_MAY_WAKE_DEVICE,
      .domain = field->domain,
      .metric_id = id,
      .instance_scope = AMDF_TELEMETRY_INSTANCE_SCOPE_DEVICE,
      .kind = field->kind,
      .format = {.kind = field->value_kind,
                 .unit = field->unit,
                 .scale_numerator = 1,
                 .scale_denominator = field->denominator,
                 .semantic = AMDF_OBSERVATION_VALUE_SEMANTIC_UNSPECIFIED},
      .name = field->name,
      .origin = {.interface_name = "linux.amdgpu.hwmon",
                 .schema = "hwmon-abi:discrete",
                 .field = field->name,
                 .field_index = UINT32_MAX},
  };
  if (profile == AMDF_GPU_KFD_HWMON_PROFILE_APU ||
      profile == AMDF_GPU_KFD_HWMON_PROFILE_RENOIR ||
      profile == AMDF_GPU_KFD_HWMON_PROFILE_SMU14_APU) {
    info.origin.schema = "hwmon-abi:apu";
    if (ordinal == 2 || ordinal == 3) {
      // APU producers report CurrentSocketPower/SocketPower, the firmware's
      // PPT/STAPM power group that can include a participating discrete GPU.
      info.domain = AMDF_TELEMETRY_DOMAIN_POWER_BUDGET;
    } else if (ordinal == 4 || ordinal == 5) {
      // The native EDGE/HOTSPOT routes return GfxTemperature/SocTemperature.
      info.kind = AMDF_TELEMETRY_METRIC_TEMPERATURE;
      info.domain =
          ordinal == 4 ? AMDF_TELEMETRY_DOMAIN_GPU : AMDF_TELEMETRY_DOMAIN_SOC;
    }
  }
  if (profile == AMDF_GPU_KFD_HWMON_PROFILE_RENOIR) {
    info.origin.schema = "hwmon-abi:renoir";
    if (ordinal == 1) {
      // This producer's GFX_MCLK route reads ClockFrequency[CLOCK_FCLK].
      info.kind = AMDF_TELEMETRY_METRIC_CLOCK_FABRIC;
      info.domain = AMDF_TELEMETRY_DOMAIN_SOC;
    }
  } else if (profile == AMDF_GPU_KFD_HWMON_PROFILE_SMU14_APU) {
    info.origin.schema = "hwmon-abi:smu14-apu";
    if (ordinal == 2 || ordinal == 3) {
      info.format.semantic = AMDF_OBSERVATION_VALUE_SEMANTIC_FILTERED;
    }
  } else if (profile == AMDF_GPU_KFD_HWMON_PROFILE_MULTI_DIE ||
             profile == AMDF_GPU_KFD_HWMON_PROFILE_MULTI_DIE_NO_GFXCLK) {
    info.origin.schema = "hwmon-abi:multi-die";
    if (ordinal == 5) {
      // MaxSocketTemperature is socket-scoped, not one graphics die.
      info.domain = AMDF_TELEMETRY_DOMAIN_PACKAGE;
    }
  }
  return info;
}

amdf_status_t amdf_gpu_umd_enumerate_telemetry_metrics(
    amdf_gpu_umd_device_t* device, amdf_platform_endpoint_t* endpoint,
    uint32_t capacity, amdf_telemetry_metric_info_t* metrics,
    uint32_t* out_count) {
  (void)device;
  int directory = -1;
  amdf_gpu_kfd_hwmon_profile_t profile;
  amdf_status_t status =
      amdf_gpu_kfd_hwmon_open(endpoint, &directory, &profile);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    *out_count = 0;
    return AMDF_STATUS_OK;
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  uint32_t supported = 0;
  const uint32_t admitted = amdf_gpu_kfd_hwmon_fields_for_profile(profile);
  for (uint32_t i = 0;
       amdf_status_is_ok(status) && i < AMDF_GPU_KFD_HWMON_FIELD_COUNT; ++i) {
    if ((admitted & (1u << i)) == 0) {
      continue;
    }
    const amdf_gpu_kfd_hwmon_field_t* field = &amdf_gpu_kfd_hwmon_fields[i];
    amdf_observation_value_t value;
    status = amdf_gpu_kfd_hwmon_read_integer(directory, field->name,
                                             field->value_kind, &value);
    if (amdf_status_is_ok(status) || status == amdf_linux_error(ENODATA)) {
      supported |= 1u << i;
      status = AMDF_STATUS_OK;
    } else if (status == amdf_linux_error(ENOENT) ||
               status == amdf_linux_error(EOPNOTSUPP)) {
      status = AMDF_STATUS_OK;
    }
  }
  const amdf_status_t close_status = amdf_linux_file_close(&directory);
  if (amdf_status_is_ok(status)) {
    status = close_status;
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  uint32_t count = 0;
  for (uint32_t i = 0; i < AMDF_GPU_KFD_HWMON_FIELD_COUNT; ++i) {
    if ((supported & (1u << i)) != 0) {
      if (count < capacity) {
        metrics[count] = amdf_gpu_kfd_hwmon_describe(profile, i);
      }
      ++count;
    }
  }
  *out_count = count;
  return capacity != 0 && capacity < count
             ? amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL)
             : AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_read_telemetry(
    amdf_gpu_umd_device_t* device, amdf_platform_endpoint_t* endpoint,
    amdf_telemetry_source_id_t source_id, uint32_t metric_count,
    const amdf_telemetry_metric_id_t* metric_ids,
    amdf_telemetry_value_t* values, amdf_telemetry_sample_t* out_sample) {
  (void)device;
  const uint64_t source_index = source_id - AMDF_GPU_KFD_HWMON_SOURCE_BASE;
  const amdf_gpu_kfd_hwmon_profile_t requested_profile =
      (amdf_gpu_kfd_hwmon_profile_t)(source_index >> 8);
  const uint32_t field_index = (uint32_t)(source_index & 0xff);
  if (source_index >> 32 != 0 || field_index == 0 ||
      field_index > AMDF_GPU_KFD_HWMON_FIELD_COUNT ||
      (amdf_gpu_kfd_hwmon_fields_for_profile(requested_profile) &
       (1u << (field_index - 1))) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }
  for (uint32_t i = 0; i < metric_count; ++i) {
    if (metric_ids[i] != source_id) {
      return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
    }
  }
  const uint32_t ordinal = field_index - 1;
  int directory = -1;
  amdf_gpu_kfd_hwmon_profile_t profile;
  amdf_status_t status =
      amdf_gpu_kfd_hwmon_open(endpoint, &directory, &profile);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_telemetry_sample_t sample = {0};
  amdf_telemetry_value_t value = {0};
  if (profile != requested_profile) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_platform_observation_query_host_clock(&sample.host_before);
  }
  if (amdf_status_is_ok(status)) {
    const amdf_gpu_kfd_hwmon_field_t* field =
        &amdf_gpu_kfd_hwmon_fields[ordinal];
    status = amdf_gpu_kfd_hwmon_read_integer(directory, field->name,
                                             field->value_kind, &value.value);
    if (status == amdf_linux_error(ENODATA) ||
        status == amdf_linux_error(EOPNOTSUPP) ||
        status == amdf_linux_error(ENOENT)) {
      value.status = amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
      status = AMDF_STATUS_OK;
    }
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_platform_observation_query_host_clock(&sample.host_after);
  }
  const amdf_status_t close_status = amdf_linux_file_close(&directory);
  if (amdf_status_is_ok(status)) {
    status = close_status;
  }
  if (amdf_status_is_ok(status)) {
    for (uint32_t i = 0; i < metric_count; ++i) {
      values[i] = value;
    }
    *out_sample = sample;
  }
  return status;
}
