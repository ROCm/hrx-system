// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/gpu/umd/kfd/topology.h"

#include <dirent.h>
#include <drm/amdgpu_drm.h>
#include <fcntl.h>
#include <linux/kfd_sysfs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/utsname.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/platform/linux/file.h"

static amdf_status_t amdf_gpu_kfd_read_attribute(int directory,
                                                 const char* name, char* text,
                                                 size_t capacity) {
  // Every exit leaves a valid empty or NUL-terminated string.
  text[0] = 0;
  int descriptor = openat(directory, name, O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return amdf_linux_error(errno);
  }
  ssize_t length;
  do {
    length = read(descriptor, text, capacity - 1);
  } while (length < 0 && errno == EINTR);
  amdf_status_t status = length < 0 ? amdf_linux_error(errno) : AMDF_STATUS_OK;
  if (length >= 0) {
    text[length] = 0;
    if ((size_t)length == capacity - 1) {
      status = amdf_linux_error(EOVERFLOW);
    }
  }
  const amdf_status_t close_status = amdf_linux_file_close(&descriptor);
  return amdf_status_is_ok(close_status) ? status : close_status;
}

static amdf_status_t amdf_gpu_kfd_read_number64(int directory, const char* name,
                                                uint64_t* out_value) {
  char text[32];
  const amdf_status_t status =
      amdf_gpu_kfd_read_attribute(directory, name, text, sizeof(text));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  char* end = NULL;
  errno = 0;
  const unsigned long long value = strtoull(text, &end, 10);
  if (errno || text[0] < '0' || text[0] > '9' || end == text ||
      (*end != 0 && (*end != '\n' || end[1] != 0))) {
    return amdf_linux_error(EPROTO);
  }
  *out_value = (uint64_t)value;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_kfd_read_number(int directory, const char* name,
                                              uint32_t* out_value) {
  uint64_t value = 0;
  const amdf_status_t status =
      amdf_gpu_kfd_read_number64(directory, name, &value);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (value > UINT32_MAX) {
    return amdf_linux_error(EPROTO);
  }
  *out_value = (uint32_t)value;
  return AMDF_STATUS_OK;
}

// Linux f145d5b3f740 and its stable backports align shader-rung SDMA
// doorbells. Release lines without that backport remain affected regardless
// of their patch number. Vendor modules have a separate patch history.
static bool amdf_gpu_kfd_kernel_has_sdma_doorbell_alignment(
    const char* release) {
  uint32_t version[3];
  const char* cursor = release;
  for (size_t i = 0; i < 3; ++i) {
    char* end = NULL;
    errno = 0;
    const unsigned long value = strtoul(cursor, &end, 10);
    if (cursor[0] < '0' || cursor[0] > '9' || errno || value > UINT32_MAX ||
        (i < 2 && *end != '.')) {
      return false;
    }
    version[i] = (uint32_t)value;
    cursor = i < 2 ? end + 1 : end;
  }
  if (*cursor != 0 && *cursor != '-' && *cursor != '+') {
    return false;
  }
  if (version[0] > 7 || (version[0] == 7 && version[1] > 3)) {
    return true;
  }
  if (version[0] == 7 && version[1] == 3) {
    if (version[2] != 0 || strncmp(cursor, "-rc", 3) != 0) {
      return true;
    }
    char* end = NULL;
    errno = 0;
    const unsigned long candidate = strtoul(cursor + 3, &end, 10);
    return cursor[3] >= '0' && cursor[3] <= '9' && !errno && candidate >= 7 &&
           (*end == 0 || *end == '-' || *end == '+');
  }
  uint32_t first_fixed_patch = UINT32_MAX;
  if (version[0] == 7 && version[1] == 2) {
    first_fixed_patch = 11;
  } else if (version[0] == 6) {
    switch (version[1]) {
      case 6:
        first_fixed_patch = 160;
        break;
      case 12:
        first_fixed_patch = 113;
        break;
      case 18:
        first_fixed_patch = 57;
        break;
    }
  }
  return first_fixed_patch != UINT32_MAX &&
         (version[2] > first_fixed_patch ||
          (version[2] == first_fixed_patch && strncmp(cursor, "-rc", 3) != 0));
}

static amdf_status_t amdf_gpu_kfd_read_sdma_device_publication(
    const amdf_platform_endpoint_t* endpoint,
    amdf_gpu_kfd_topology_t* topology) {
  const amdf_gpu_kfd_ip_version_t gc = topology->gc_ip;
  const amdf_gpu_endpoint_properties_t* properties = &topology->properties;
  // Compiler targets gfx940-942 alias GC9.4.3/9.4.4. Missing exact discovery
  // must not admit the same affected devices through their compiler alias.
  topology->sdma.device_producer_restricted =
      gc.exact
          ? gc.major == 9 &&
                ((gc.minor == 4 && (gc.revision == 3 || gc.revision == 4)) ||
                 (gc.minor == 5 && gc.revision == 0))
          : properties->gfx_ip.major == 9 &&
                ((properties->gfx_ip.minor == 4 &&
                  properties->gfx_ip.stepping <= 2) ||
                 (properties->gfx_ip.minor == 5 &&
                  properties->gfx_ip.stepping == 0));
  if (!topology->sdma.device_producer_restricted) {
    return AMDF_STATUS_OK;
  }
  char version[64];
  const amdf_status_t status = amdf_gpu_kfd_read_attribute(
      endpoint->instance->sysfs_descriptor, "module/amdgpu/version", version,
      sizeof(version));
  if (amdf_status_is_ok(status) || status == amdf_linux_error(EACCES) ||
      status == amdf_linux_error(EOVERFLOW)) {
    // Upstream amdgpu has no MODULE_VERSION. Packaged drivers do, and no
    // published vendor release is qualified with the alignment repair.
    // Missing or inaccessible identity is not evidence of a fixed driver.
    return AMDF_STATUS_OK;
  }
  if (status != amdf_linux_error(ENOENT)) {
    return status;
  }
  char taint[64];
  const amdf_status_t taint_status =
      amdf_gpu_kfd_read_attribute(endpoint->instance->sysfs_descriptor,
                                  "module/amdgpu/taint", taint, sizeof(taint));
  if (taint_status == amdf_linux_error(ENOENT) ||
      taint_status == amdf_linux_error(EACCES) ||
      taint_status == amdf_linux_error(EOVERFLOW)) {
    return AMDF_STATUS_OK;
  }
  if (!amdf_status_is_ok(taint_status)) {
    return taint_status;
  }
  if (strchr(taint, 'O') != NULL) {
    return AMDF_STATUS_OK;
  }
  struct utsname kernel;
  if (uname(&kernel) != 0) {
    return amdf_linux_error(errno);
  }
  topology->sdma.device_producer_restricted =
      !amdf_gpu_kfd_kernel_has_sdma_doorbell_alignment(kernel.release);
  return AMDF_STATUS_OK;
}

// KFD installs the CWSR trap in the process context, independently of the
// compiler descriptor. Its read-only module parameter determines the extra
// SGPR reservation on the CDNA devices admitted by our compute queue plans.
static amdf_status_t amdf_gpu_kfd_read_scalar_register_features(
    const amdf_platform_endpoint_t* endpoint,
    amdf_gpu_kfd_topology_t* topology) {
  const amdf_gpu_endpoint_properties_t* properties = &topology->properties;
  if (properties->gfx_ip.major != 9 ||
      !((properties->gfx_ip.minor == 0 && properties->gfx_ip.stepping == 10) ||
        properties->gfx_ip.minor == 4 || properties->gfx_ip.minor == 5)) {
    return AMDF_STATUS_OK;
  }
  uint32_t enabled = 0;
  const amdf_status_t status = amdf_gpu_kfd_read_number(
      endpoint->instance->sysfs_descriptor,
      "module/amdgpu/parameters/cwsr_enable", &enabled);
  if (status == amdf_linux_error(ENOENT) ||
      status == amdf_linux_error(EACCES)) {
    // A restricted sysfs namespace does not remove native queue support.
    // It only prevents an exact answer about this optional context fact.
    return AMDF_STATUS_OK;
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  topology->aql_scalar_register_features =
      enabled ? AMDF_GPU_AQL_FORMAT_FEATURE_SGPR_TRAP_RESERVE_16
              : AMDF_GPU_AQL_FORMAT_FEATURE_SGPR_TRAP_RESERVE_0;
  return AMDF_STATUS_OK;
}

// KFD topology properties are decimal name/value records. Baseline endpoint
// fields are mandatory, including fields for which zero is meaningful. Compute
// storage and SDMA groups are independently optional on older kernels, but each
// group must be complete when present.
static amdf_status_t amdf_gpu_kfd_read_node(
    int directory, const amdf_platform_endpoint_t* endpoint,
    amdf_gpu_kfd_topology_t* topology, uint32_t* out_io_link_count,
    bool* out_matches) {
  enum {
    RENDER_MINOR,
    VENDOR_ID,
    DEVICE_ID,
    GFX_TARGET,
    CAPABILITY,
    SIMD_COUNT,
    SIMD_PER_CU,
    WAVES_PER_SIMD,
    SCRATCH_SLOTS,
    WAVE_SIZE,
    LDS_KIB,
    ARRAY_COUNT,
    ARRAYS_PER_ENGINE,
    XCC_COUNT,
    COMPUTE_QUEUE_COUNT,
    P2P_LINK_COUNT,
    REQUIRED_PROPERTY_COUNT,
    CONTEXT_SAVE_RESTORE_SIZE = REQUIRED_PROPERTY_COUNT,
    CONTROL_STACK_SIZE,
    SDMA_ENGINE_COUNT,
    SDMA_XGMI_ENGINE_COUNT,
    SDMA_QUEUE_COUNT_PER_ENGINE,
    IO_LINK_COUNT,
    GLOBAL_WAVE_SYNC_COUNT,
    PROPERTY_COUNT,
  };
  const char* names[PROPERTY_COUNT] = {
      "drm_render_minor",
      "vendor_id",
      "device_id",
      "gfx_target_version",
      "capability",
      "simd_count",
      "simd_per_cu",
      "max_waves_per_simd",
      "max_slots_scratch_cu",
      "wave_front_size",
      "lds_size_in_kb",
      "array_count",
      "simd_arrays_per_engine",
      "num_xcc",
      "num_cp_queues",
      "p2p_links_count",
      "cwsr_size",
      "ctl_stack_size",
      "num_sdma_engines",
      "num_sdma_xgmi_engines",
      "num_sdma_queues_per_engine",
      "io_links_count",
      "num_gws",
  };
  uint32_t values[PROPERTY_COUNT] = {0};
  uint32_t present = 0;
  uint64_t unique_id = 0;
  bool unique_id_present = false;
  char text[8192];
  amdf_status_t status =
      amdf_gpu_kfd_read_attribute(directory, "properties", text, sizeof(text));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  char* cursor = text;
  while (*cursor != 0) {
    char* separator = strchr(cursor, ' ');
    if (separator == NULL) {
      return amdf_linux_error(EPROTO);
    }
    *separator = 0;
    char* end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(separator + 1, &end, 10);
    if (errno || separator[1] < '0' || separator[1] > '9' ||
        end == separator + 1 || *end != '\n') {
      return amdf_linux_error(EPROTO);
    }
    if (strcmp(cursor, "unique_id") == 0) {
      if (unique_id_present) {
        return amdf_linux_error(EPROTO);
      }
      unique_id = (uint64_t)value;
      unique_id_present = true;
    }
    for (uint32_t i = 0; i < PROPERTY_COUNT; ++i) {
      if (strcmp(cursor, names[i]) != 0) {
        continue;
      }
      if (value > UINT32_MAX || (present & (1u << i)) != 0) {
        return amdf_linux_error(EPROTO);
      }
      values[i] = (uint32_t)value;
      present |= 1u << i;
      break;
    }
    cursor = end + 1;
  }
  if ((present & (1u << RENDER_MINOR)) == 0) {
    return amdf_linux_error(EPROTO);
  }
  if (values[RENDER_MINOR] != (uint32_t)endpoint->info.id.words[0]) {
    return AMDF_STATUS_OK;
  }
  const uint32_t required_properties = (1u << REQUIRED_PROPERTY_COUNT) - 1;
  const uint32_t compute_storage_properties =
      (1u << CONTEXT_SAVE_RESTORE_SIZE) | (1u << CONTROL_STACK_SIZE);
  const uint32_t sdma_properties = (1u << SDMA_ENGINE_COUNT) |
                                   (1u << SDMA_XGMI_ENGINE_COUNT) |
                                   (1u << SDMA_QUEUE_COUNT_PER_ENGINE);
  if ((present & required_properties) != required_properties ||
      ((present & compute_storage_properties) != 0 &&
       (present & compute_storage_properties) != compute_storage_properties) ||
      ((present & sdma_properties) != 0 &&
       (present & sdma_properties) != sdma_properties)) {
    return amdf_linux_error(EPROTO);
  }
  if (values[VENDOR_ID] != endpoint->info.pci.vendor_id ||
      values[DEVICE_ID] != endpoint->info.pci.device_id) {
    return amdf_linux_error(ENODEV);
  }
  const uint64_t arrays_per_xcc =
      (uint64_t)values[ARRAYS_PER_ENGINE] * values[XCC_COUNT];
  const uint64_t maximum_waves =
      (uint64_t)values[WAVES_PER_SIMD] * values[SIMD_PER_CU];
  if (values[SIMD_PER_CU] == 0 ||
      values[SIMD_COUNT] % values[SIMD_PER_CU] != 0 || arrays_per_xcc == 0 ||
      values[ARRAY_COUNT] % arrays_per_xcc != 0 || maximum_waves > UINT32_MAX) {
    return amdf_linux_error(EPROTO);
  }
  topology->properties = (amdf_gpu_endpoint_properties_t){
      .gfx_ip =
          {
              .major = values[GFX_TARGET] / 10000,
              .minor = (values[GFX_TARGET] / 100) % 100,
              .stepping = values[GFX_TARGET] % 100,
          },
      .asic_revision = (values[CAPABILITY] & HSA_CAP_ASIC_REVISION_MASK) >>
                       HSA_CAP_ASIC_REVISION_SHIFT,
      .unique_id = unique_id,
      .compute =
          {
              .wavefront_size = values[WAVE_SIZE],
              .compute_unit_count = values[SIMD_COUNT] / values[SIMD_PER_CU],
              .maximum_wave_count_per_compute_unit = (uint32_t)maximum_waves,
              .maximum_scratch_wave_count_per_compute_unit =
                  values[SCRATCH_SLOTS],
              .local_data_share_byte_length = (uint64_t)values[LDS_KIB] * 1024,
          },
      .topology =
          {
              .xcc_count = values[XCC_COUNT],
              .shader_engine_count_per_xcc =
                  (uint32_t)(values[ARRAY_COUNT] / arrays_per_xcc),
          },
  };
  topology->compute_queue_count = values[COMPUTE_QUEUE_COUNT];
  topology->global_wave_sync_count = values[GLOBAL_WAVE_SYNC_COUNT];
  topology->memory_peers.count = values[P2P_LINK_COUNT];
  topology->sdma.engine_count = values[SDMA_ENGINE_COUNT];
  topology->sdma.xgmi_engine_count = values[SDMA_XGMI_ENGINE_COUNT];
  topology->sdma.queue_count_per_engine = values[SDMA_QUEUE_COUNT_PER_ENGINE];
  topology->context_save_restore_byte_length =
      values[CONTEXT_SAVE_RESTORE_SIZE];
  topology->control_stack_byte_length = values[CONTROL_STACK_SIZE];
  *out_io_link_count = values[IO_LINK_COUNT];
  *out_matches = true;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_kfd_read_ip_version(
    const amdf_platform_endpoint_t* endpoint, uint32_t hardware_id,
    amdf_gpu_kfd_ip_version_t* out_version) {
  // Numeric hardware IDs also work before sysfs added named IP symlinks.
  // Discovery distinguishes the native revision from compiler target aliases.
  char path[128];
  snprintf(path, sizeof(path), "dev/char/%u:%u/device/ip_discovery/die/0/%u/0",
           (uint32_t)(endpoint->info.id.words[0] >> 32),
           (uint32_t)endpoint->info.id.words[0], hardware_id);
  int directory = openat(endpoint->instance->sysfs_descriptor, path,
                         O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (directory < 0) {
    // A provider without discovery metadata cannot select an exact packet ABI.
    // Missing fields inside a present record, however, are errors.
    return errno == ENOENT ? AMDF_STATUS_OK : amdf_linux_error(errno);
  }
  const char* attributes[] = {"major", "minor", "revision"};
  uint32_t values[3] = {0};
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0; i < 3 && amdf_status_is_ok(status); ++i) {
    status = amdf_gpu_kfd_read_number(directory, attributes[i], &values[i]);
    if (amdf_status_is_ok(status) && values[i] > UINT8_MAX) {
      status = amdf_linux_error(EPROTO);
    }
  }
  const amdf_status_t close_status = amdf_linux_file_close(&directory);
  if (!amdf_status_is_ok(close_status)) {
    status = close_status;
  }
  if (amdf_status_is_ok(status) && (values[0] | values[1] | values[2]) != 0) {
    *out_version = (amdf_gpu_kfd_ip_version_t){
        .major = values[0],
        .minor = values[1],
        .revision = values[2],
        .exact = true,
    };
  }
  return status;
}

static amdf_status_t amdf_gpu_kfd_read_memory(
    const amdf_platform_endpoint_t* endpoint,
    amdf_gpu_kfd_topology_t* topology) {
  // These sysfs totals expose the same initialized heap capacities as
  // INFO_MEMORY, without sampling current usage or allocating payload.
  char path[64];
  snprintf(path, sizeof(path), "dev/char/%u:%u/device",
           (uint32_t)(endpoint->info.id.words[0] >> 32),
           (uint32_t)endpoint->info.id.words[0]);
  int directory = openat(endpoint->instance->sysfs_descriptor, path,
                         O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (directory < 0) {
    return amdf_linux_error(errno);
  }
  uint64_t total_vram = 0;
  uint64_t visible_vram = 0;
  uint64_t total_gtt = 0;
  amdf_status_t status =
      amdf_gpu_kfd_read_number64(directory, "mem_info_vram_total", &total_vram);
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_read_number64(directory, "mem_info_vis_vram_total",
                                        &visible_vram);
  }
  if (amdf_status_is_ok(status)) {
    status =
        amdf_gpu_kfd_read_number64(directory, "mem_info_gtt_total", &total_gtt);
  }
  uint64_t hive_id = 0;
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_read_number64(
        directory, "xgmi_hive_info/xgmi_hive_id", &hive_id);
    if (status == amdf_linux_error(ENOENT)) {
      status = AMDF_STATUS_OK;
    }
  }
  uint32_t hive_sharing_enabled = 0;
  if (amdf_status_is_ok(status) && hive_id != 0) {
    status = amdf_gpu_kfd_read_number(endpoint->instance->sysfs_descriptor,
                                      "module/amdgpu/parameters/use_xgmi_p2p",
                                      &hive_sharing_enabled);
  }
  const amdf_status_t close_status = amdf_linux_file_close(&directory);
  if (!amdf_status_is_ok(close_status)) {
    status = close_status;
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  topology->vram.total_byte_length = total_vram;
  topology->vram.visible_byte_length = visible_vram;
  topology->gtt_byte_length = total_gtt;
  topology->memory_peers.hive_id = hive_id;
  topology->memory_peers.hive_sharing_enabled = hive_sharing_enabled != 0;
  return AMDF_STATUS_OK;
}

// One directed native topology edge.
typedef struct amdf_gpu_kfd_link_t {
  // Native interconnect class, independent of the endpoint's compiler target.
  uint32_t type;
  // Destination KFD topology node ordinal.
  uint32_t destination;
  // Driver-reported reachability, coherence and atomic width flags.
  uint32_t flags;
} amdf_gpu_kfd_link_t;

static amdf_status_t amdf_gpu_kfd_read_link(int node, uint32_t node_ordinal,
                                            const char* directory,
                                            uint32_t link_ordinal,
                                            amdf_gpu_kfd_link_t* out_link) {
  char path[64];
  snprintf(path, sizeof(path), "%s/%u/properties", directory, link_ordinal);
  char text[2048];
  const amdf_status_t status =
      amdf_gpu_kfd_read_attribute(node, path, text, sizeof(text));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const char* names[] = {"type", "node_from", "node_to", "flags"};
  uint32_t values[4] = {0};
  uint32_t present = 0;
  char* cursor = text;
  while (*cursor != 0) {
    char* separator = strchr(cursor, ' ');
    if (separator == NULL) {
      return amdf_linux_error(EPROTO);
    }
    *separator = 0;
    char* end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(separator + 1, &end, 10);
    if (errno || separator[1] < '0' || separator[1] > '9' ||
        end == separator + 1 || *end != '\n') {
      return amdf_linux_error(EPROTO);
    }
    for (uint32_t i = 0; i < 4; ++i) {
      if (strcmp(cursor, names[i]) != 0) {
        continue;
      }
      if (value > UINT32_MAX || (present & (1u << i)) != 0) {
        return amdf_linux_error(EPROTO);
      }
      values[i] = (uint32_t)value;
      present |= 1u << i;
      break;
    }
    cursor = end + 1;
  }
  if (present != 15 || values[1] != node_ordinal) {
    return amdf_linux_error(EPROTO);
  }
  *out_link = (amdf_gpu_kfd_link_t){
      .type = values[0], .destination = values[2], .flags = values[3]};
  return AMDF_STATUS_OK;
}

// KFD publishes an indirect peer edge only after the backing owner's BAR,
// consumer DMA aperture and platform P2P admission checks. The edge is stored
// on the consumer and points to the backing node; the reverse is independent.
static amdf_status_t amdf_gpu_kfd_read_memory_peers(
    int nodes, int node, uint32_t node_ordinal, amdf_allocator_t host_allocator,
    amdf_gpu_kfd_topology_t* topology) {
  const uint32_t link_count = topology->memory_peers.count;
  topology->memory_peers.count = 0;
  if (link_count == 0) {
    return AMDF_STATUS_OK;
  }
  amdf_status_t status = amdf_malloc(
      host_allocator, (uint64_t)link_count * sizeof(uint32_t),
      amdf_alignof(uint32_t), (void**)&topology->memory_peers.gpu_ids);
  for (uint32_t i = 0; amdf_status_is_ok(status) && i < link_count; ++i) {
    amdf_gpu_kfd_link_t link = {0};
    status = amdf_gpu_kfd_read_link(node, node_ordinal, "p2p_links", i, &link);
    if (!amdf_status_is_ok(status) || link.type != HSA_IOLINK_TYPE_PCIEXPRESS ||
        (link.flags &
         (HSA_IOLINK_FLAGS_ENABLED | HSA_IOLINK_FLAGS_NO_PEER_TO_PEER_DMA)) !=
            HSA_IOLINK_FLAGS_ENABLED) {
      continue;
    }
    char path[64];
    snprintf(path, sizeof(path), "%u/gpu_id", link.destination);
    uint32_t gpu_id = 0;
    status = amdf_gpu_kfd_read_number(nodes, path, &gpu_id);
    if (status == amdf_linux_error(EPERM)) {
      // Backing outside the process's device cgroup is not a peer route.
      status = AMDF_STATUS_OK;
    } else if (amdf_status_is_ok(status) && gpu_id != 0) {
      topology->memory_peers.gpu_ids[topology->memory_peers.count++] = gpu_id;
    }
  }
  return status;
}

// System allocations are not bound to a selected CPU NUMA node. Intersect
// the widths of every reported CPU route instead of choosing the first link.
static amdf_status_t amdf_gpu_kfd_read_host_access(
    int nodes, int node, uint32_t node_ordinal, uint32_t link_count,
    amdf_gpu_kfd_topology_t* topology) {
  uint32_t cpu_link_count = 0;
  bool supports_32 = true;
  bool supports_64 = true;
  bool host_cached_vram = false;
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0; amdf_status_is_ok(status) && i < link_count; ++i) {
    amdf_gpu_kfd_link_t link = {0};
    status = amdf_gpu_kfd_read_link(node, node_ordinal, "io_links", i, &link);
    uint32_t gpu_id = 0;
    if (amdf_status_is_ok(status)) {
      char path[64];
      snprintf(path, sizeof(path), "%u/gpu_id", link.destination);
      status = amdf_gpu_kfd_read_number(nodes, path, &gpu_id);
      if (status == amdf_linux_error(EPERM)) {
        // CPU nodes remain visible. A denied GPU is not a zero-ID CPU route.
        status = AMDF_STATUS_OK;
        continue;
      }
    }
    if (amdf_status_is_ok(status) && gpu_id == 0) {
      ++cpu_link_count;
      host_cached_vram |= link.type == HSA_IOLINK_TYPE_XGMI &&
                          (link.flags & HSA_IOLINK_FLAGS_ENABLED) != 0;
      supports_32 &= (link.flags & (HSA_IOLINK_FLAGS_ENABLED |
                                    HSA_IOLINK_FLAGS_NO_ATOMICS_32_BIT)) ==
                     HSA_IOLINK_FLAGS_ENABLED;
      supports_64 &= (link.flags & (HSA_IOLINK_FLAGS_ENABLED |
                                    HSA_IOLINK_FLAGS_NO_ATOMICS_64_BIT)) ==
                     HSA_IOLINK_FLAGS_ENABLED;
    }
  }
  if (amdf_status_is_ok(status)) {
    topology->host_atomics.supports_32 = cpu_link_count != 0 && supports_32;
    topology->host_atomics.supports_64 = cpu_link_count != 0 && supports_64;
    topology->vram.host_cached = host_cached_vram;
  }
  return status;
}

void amdf_gpu_kfd_topology_deinitialize(amdf_gpu_kfd_topology_t* topology,
                                        amdf_allocator_t host_allocator) {
  amdf_free(host_allocator, topology->memory_peers.gpu_ids);
  topology->memory_peers.gpu_ids = NULL;
  topology->memory_peers.count = 0;
}

amdf_gpu_device_features_t amdf_gpu_kfd_topology_memory_features(
    const amdf_gpu_kfd_topology_t* topology, bool integrated) {
  // The supported KFD allocator selects apu_prefer_gtt at initialization when
  // an APU's real VRAM is smaller than its GTT manager. In that configuration
  // VRAM requests are redirected before allocation, so no LOCAL route exists.
  // Equal totals retain VRAM. Later eviction is independent of this admission.
  if (topology->vram.total_byte_length == 0 ||
      (integrated &&
       topology->vram.total_byte_length < topology->gtt_byte_length)) {
    return 0;
  }
  amdf_gpu_device_features_t features = AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY;
  if (topology->vram.visible_byte_length >= topology->vram.total_byte_length) {
    features |= AMDF_GPU_DEVICE_FEATURE_HOST_VISIBLE_LOCAL_MEMORY;
  }
  return features;
}

amdf_status_t amdf_gpu_kfd_topology_query_memory(
    int render_descriptor, uint32_t pci_device_id,
    amdf_gpu_kfd_topology_t* topology) {
  struct drm_amdgpu_info_device device = {0};
  struct drm_amdgpu_info query = {
      .return_pointer = (uintptr_t)&device,
      .return_size = sizeof(device),
      .query = AMDGPU_INFO_DEV_INFO,
  };
  if (ioctl(render_descriptor, DRM_IOCTL_AMDGPU_INFO, &query) != 0) {
    return amdf_linux_error(errno);
  }
  if (device.device_id != pci_device_id ||
      device.virtual_address_offset >= device.virtual_address_max ||
      device.virtual_address_alignment == 0 ||
      (device.virtual_address_alignment &
       (device.virtual_address_alignment - 1)) != 0 ||
      device.pte_fragment_size == 0 ||
      (device.pte_fragment_size & (device.pte_fragment_size - 1)) != 0) {
    return amdf_linux_error(EPROTO);
  }
  topology->virtual_address.begin = device.virtual_address_offset;
  topology->virtual_address.end = device.virtual_address_max;
  topology->virtual_address.alignment = device.virtual_address_alignment;
  topology->virtual_address.fragment_size = device.pte_fragment_size;
  topology->integrated = (device.ids_flags & AMDGPU_IDS_FLAGS_FUSION) != 0;
  topology->virtual_function =
      ((device.ids_flags & AMDGPU_IDS_FLAGS_MODE_MASK) >>
       AMDGPU_IDS_FLAGS_MODE_SHIFT) == AMDGPU_IDS_FLAGS_MODE_VF;
  topology->memory_features =
      amdf_gpu_kfd_topology_memory_features(topology, topology->integrated);
  return AMDF_STATUS_OK;
}

amdf_gpu_execution_resource_geometry_t
amdf_gpu_kfd_topology_execution_resources(
    const amdf_gpu_kfd_topology_t* topology) {
  const amdf_gpu_endpoint_properties_t* properties = &topology->properties;
  const uint32_t unit_count = properties->compute.compute_unit_count;
  const uint32_t group_count = properties->topology.xcc_count;
  const uint32_t major = properties->gfx_ip.major;
  const uint32_t minor = properties->gfx_ip.minor;
  const uint32_t stepping = properties->gfx_ip.stepping;
  const bool cdna = major == 9 && ((minor == 0 && stepping == 10) ||
                                   (minor == 4 && stepping <= 2) ||
                                   (minor == 5 && stepping == 0));
  const bool paired = major == 11 || (major == 12 && minor == 0);
  // GC 12.1 maps one logical bit to a native pair; its reported compute-unit
  // count already uses that unit. Compiler GFX125x alone does not identify the
  // installed native mapper, so require its exact discovery identity.
  const bool gc121 = major == 12 && minor == 5 && topology->gc_ip.exact &&
                     topology->gc_ip.major == 12 &&
                     topology->gc_ip.minor == 1 &&
                     topology->gc_ip.revision == 0;
  if ((!cdna && !paired && !gc121) || unit_count == 0 || unit_count > 1024 ||
      group_count == 0 || (paired && group_count != 1)) {
    return (amdf_gpu_execution_resource_geometry_t){0};
  }
  const uint32_t units_per_resource = paired ? 2 : 1;
  if (unit_count % units_per_resource != 0 ||
      (unit_count / units_per_resource) % group_count != 0) {
    return (amdf_gpu_execution_resource_geometry_t){0};
  }
  // Each command processor retains at least one enabled resource. KFD maps
  // interleaved logical mask positions onto enabled units after harvesting.
  return (amdf_gpu_execution_resource_geometry_t){
      .compute_units_per_resource = units_per_resource,
      .group_count = group_count,
      .minimum_selected_resources_per_group = 1,
  };
}

amdf_status_t amdf_gpu_kfd_topology_initialize(
    const amdf_platform_endpoint_t* endpoint, amdf_allocator_t host_allocator,
    amdf_gpu_kfd_topology_t* out_topology) {
  int topology_directory =
      openat(endpoint->instance->sysfs_descriptor, "class/kfd/kfd/topology",
             O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (topology_directory < 0) {
    return errno == ENOENT ? amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)
                           : amdf_linux_error(errno);
  }
  uint32_t generation = 0;
  amdf_status_t status = amdf_gpu_kfd_read_number(topology_directory,
                                                  "generation_id", &generation);
  int nodes = -1;
  int nodes_directory = -1;
  DIR* directory = NULL;
  amdf_gpu_kfd_topology_t topology = {0};
  bool found = false;
  if (amdf_status_is_ok(status)) {
    nodes =
        openat(topology_directory, "nodes", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (nodes < 0) {
      status = amdf_linux_error(errno);
    } else {
      directory = fdopendir(nodes);
      if (directory == NULL) {
        status = amdf_linux_error(errno);
      } else {
        nodes = -1;
        nodes_directory = dirfd(directory);
        if (nodes_directory < 0) {
          status = amdf_linux_error(errno);
        }
      }
    }
  }
  while (directory != NULL && nodes_directory >= 0 &&
         amdf_status_is_ok(status) && !found) {
    errno = 0;
    const struct dirent* entry = readdir(directory);
    if (entry == NULL) {
      if (errno) {
        status = amdf_linux_error(errno);
      }
      break;
    }
    if (entry->d_name[0] == 0 ||
        strspn(entry->d_name, "0123456789") != strlen(entry->d_name)) {
      continue;
    }
    int node = openat(nodes_directory, entry->d_name,
                      O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (node < 0) {
      status = amdf_linux_error(errno);
      break;
    }
    uint32_t gpu_id = 0;
    status = amdf_gpu_kfd_read_number(node, "gpu_id", &gpu_id);
    if (status == amdf_linux_error(EPERM)) {
      // KFD denies gpu_id reads for GPUs excluded by the device cgroup.
      status = AMDF_STATUS_OK;
    } else if (amdf_status_is_ok(status) && gpu_id != 0) {
      uint32_t io_link_count = 0;
      status = amdf_gpu_kfd_read_node(node, endpoint, &topology, &io_link_count,
                                      &found);
      if (amdf_status_is_ok(status) && found) {
        topology.gpu_id = gpu_id;
        char* end = NULL;
        errno = 0;
        const unsigned long node_ordinal = strtoul(entry->d_name, &end, 10);
        if (errno || *end != 0 || node_ordinal > UINT32_MAX) {
          status = amdf_linux_error(EPROTO);
        } else {
          status = amdf_gpu_kfd_read_memory_peers(nodes_directory, node,
                                                  (uint32_t)node_ordinal,
                                                  host_allocator, &topology);
          if (amdf_status_is_ok(status)) {
            status = amdf_gpu_kfd_read_host_access(nodes_directory, node,
                                                   (uint32_t)node_ordinal,
                                                   io_link_count, &topology);
          }
        }
      }
    }
    const amdf_status_t close_status = amdf_linux_file_close(&node);
    if (!amdf_status_is_ok(close_status)) {
      status = close_status;
    }
  }
  if (amdf_status_is_ok(status) && found) {
    // Native hardware IDs: GC is 11 and SDMA0 is 42.
    status = amdf_gpu_kfd_read_ip_version(endpoint, 11, &topology.gc_ip);
  }
  if (amdf_status_is_ok(status) && found &&
      (topology.sdma.engine_count != 0 ||
       topology.sdma.xgmi_engine_count != 0)) {
    status = amdf_gpu_kfd_read_ip_version(endpoint, 42, &topology.sdma.ip);
  }
  if (amdf_status_is_ok(status) && found) {
    status = amdf_gpu_kfd_read_memory(endpoint, &topology);
  }
  if (amdf_status_is_ok(status) && found) {
    status = amdf_gpu_kfd_read_scalar_register_features(endpoint, &topology);
  }
  if (amdf_status_is_ok(status) && found) {
    status = amdf_gpu_kfd_read_sdma_device_publication(endpoint, &topology);
  }
  uint32_t final_generation = 0;
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_read_number(topology_directory, "generation_id",
                                      &final_generation);
    if (amdf_status_is_ok(status) && final_generation != generation) {
      status = amdf_linux_error(EAGAIN);
    }
  }
  if (directory != NULL && closedir(directory) != 0) {
    status = amdf_linux_error(errno);
  }
  const amdf_status_t nodes_status = amdf_linux_file_close(&nodes);
  if (!amdf_status_is_ok(nodes_status)) {
    status = nodes_status;
  }
  const amdf_status_t close_status = amdf_linux_file_close(&topology_directory);
  if (!amdf_status_is_ok(close_status)) {
    status = close_status;
  }
  if (amdf_status_is_ok(status) && !found) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (amdf_status_is_ok(status)) {
    topology.properties.execution_resources =
        amdf_gpu_kfd_topology_execution_resources(&topology);
    *out_topology = topology;
  } else {
    amdf_gpu_kfd_topology_deinitialize(&topology, host_allocator);
  }
  return status;
}
