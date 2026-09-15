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

#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/platform/linux/file.h"

static amdf_status_t amdf_gpu_kfd_read_attribute(int directory,
                                                 const char* name, char* text,
                                                 size_t capacity) {
  int descriptor = openat(directory, name, O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) return amdf_linux_error(errno);
  ssize_t length;
  do {
    length = read(descriptor, text, capacity - 1);
  } while (length < 0 && errno == EINTR);
  amdf_status_t status = length < 0 ? amdf_linux_error(errno) : AMDF_STATUS_OK;
  if (length >= 0) {
    text[length] = 0;
    if ((size_t)length == capacity - 1) status = amdf_linux_error(EOVERFLOW);
  }
  const amdf_status_t close_status = amdf_linux_file_close(&descriptor);
  return amdf_status_is_ok(close_status) ? status : close_status;
}

static amdf_status_t amdf_gpu_kfd_read_number64(int directory, const char* name,
                                                uint64_t* out_value) {
  char text[32];
  const amdf_status_t status =
      amdf_gpu_kfd_read_attribute(directory, name, text, sizeof(text));
  if (!amdf_status_is_ok(status)) return status;
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
  if (!amdf_status_is_ok(status)) return status;
  if (value > UINT32_MAX) return amdf_linux_error(EPROTO);
  *out_value = (uint32_t)value;
  return AMDF_STATUS_OK;
}

// KFD topology properties are decimal name/value records. Baseline endpoint
// fields are mandatory, including fields for which zero is meaningful. Compute
// storage and SDMA groups are independently optional on older kernels, but each
// group must be complete when present.
static amdf_status_t amdf_gpu_kfd_read_node(
    int directory, const amdf_platform_endpoint_t* endpoint,
    amdf_gpu_kfd_topology_t* topology, bool* out_matches) {
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
  };
  uint32_t values[PROPERTY_COUNT] = {0};
  uint32_t present = 0;
  char text[8192];
  amdf_status_t status =
      amdf_gpu_kfd_read_attribute(directory, "properties", text, sizeof(text));
  if (!amdf_status_is_ok(status)) return status;
  char* cursor = text;
  while (*cursor != 0) {
    char* separator = strchr(cursor, ' ');
    if (separator == NULL) return amdf_linux_error(EPROTO);
    *separator = 0;
    char* end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(separator + 1, &end, 10);
    if (errno || end == separator + 1 || *end != '\n') {
      return amdf_linux_error(EPROTO);
    }
    for (uint32_t i = 0; i < PROPERTY_COUNT; ++i) {
      if (strcmp(cursor, names[i]) != 0) continue;
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
  topology->memory_peers.count = values[P2P_LINK_COUNT];
  topology->sdma.engine_count = values[SDMA_ENGINE_COUNT];
  topology->sdma.xgmi_engine_count = values[SDMA_XGMI_ENGINE_COUNT];
  topology->sdma.queue_count_per_engine = values[SDMA_QUEUE_COUNT_PER_ENGINE];
  topology->context_save_restore_byte_length =
      values[CONTEXT_SAVE_RESTORE_SIZE];
  topology->control_stack_byte_length = values[CONTROL_STACK_SIZE];
  *out_matches = true;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_kfd_query_sdma(
    const amdf_platform_endpoint_t* endpoint,
    amdf_gpu_kfd_topology_t* topology) {
  if (topology->sdma.engine_count == 0 &&
      topology->sdma.xgmi_engine_count == 0) {
    return AMDF_STATUS_OK;
  }
  // Hardware ID 42 is SDMA0. These cached discovery bytes are the same version
  // returned by HW_IP_INFO for SDMA instance zero, without a native query. The
  // numeric hardware-ID path also works before sysfs added named IP symlinks.
  char path[128];
  snprintf(path, sizeof(path), "dev/char/%u:%u/device/ip_discovery/die/0/42/0",
           (uint32_t)(endpoint->info.id.words[0] >> 32),
           (uint32_t)endpoint->info.id.words[0]);
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
  if (!amdf_status_is_ok(close_status)) status = close_status;
  if (amdf_status_is_ok(status) && (values[0] | values[1] | values[2]) != 0) {
    topology->sdma.ip.major = values[0];
    topology->sdma.ip.minor = values[1];
    topology->sdma.ip.revision = values[2];
    topology->sdma.ip.exact = true;
  }
  return status;
}

static amdf_status_t amdf_gpu_kfd_read_memory(
    const amdf_platform_endpoint_t* endpoint,
    amdf_gpu_kfd_topology_t* topology) {
  // These sysfs totals format the same cached real/visible VRAM sizes used by
  // INFO_MEMORY, without issuing a native query or sampling allocation usage.
  char path[64];
  snprintf(path, sizeof(path), "dev/char/%u:%u/device",
           (uint32_t)(endpoint->info.id.words[0] >> 32),
           (uint32_t)endpoint->info.id.words[0]);
  int directory = openat(endpoint->instance->sysfs_descriptor, path,
                         O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (directory < 0) return amdf_linux_error(errno);
  uint64_t total_vram = 0;
  uint64_t visible_vram = 0;
  amdf_status_t status =
      amdf_gpu_kfd_read_number64(directory, "mem_info_vram_total", &total_vram);
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_read_number64(directory, "mem_info_vis_vram_total",
                                        &visible_vram);
  }
  uint64_t hive_id = 0;
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_read_number64(
        directory, "xgmi_hive_info/xgmi_hive_id", &hive_id);
    if (status == amdf_linux_error(ENOENT)) status = AMDF_STATUS_OK;
  }
  uint32_t hive_sharing_enabled = 0;
  if (amdf_status_is_ok(status) && hive_id != 0) {
    status = amdf_gpu_kfd_read_number(endpoint->instance->sysfs_descriptor,
                                      "module/amdgpu/parameters/use_xgmi_p2p",
                                      &hive_sharing_enabled);
  }
  const amdf_status_t close_status = amdf_linux_file_close(&directory);
  if (!amdf_status_is_ok(close_status)) status = close_status;
  if (!amdf_status_is_ok(status)) return status;
  topology->vram.total_byte_length = total_vram;
  topology->vram.visible_byte_length = visible_vram;
  topology->memory_peers.hive_id = hive_id;
  topology->memory_peers.hive_sharing_enabled = hive_sharing_enabled != 0;
  return AMDF_STATUS_OK;
}

// KFD publishes an indirect peer edge only after the backing owner's BAR,
// consumer DMA aperture and platform P2P admission checks. The edge is stored
// on the consumer and points to the backing node; the reverse is independent.
static amdf_status_t amdf_gpu_kfd_read_peer_link(int node,
                                                 uint32_t node_ordinal,
                                                 uint32_t link_ordinal,
                                                 uint32_t* out_backing_node,
                                                 bool* out_enabled) {
  char path[64];
  snprintf(path, sizeof(path), "p2p_links/%u/properties", link_ordinal);
  char text[2048];
  const amdf_status_t status =
      amdf_gpu_kfd_read_attribute(node, path, text, sizeof(text));
  if (!amdf_status_is_ok(status)) return status;
  const char* names[] = {"type", "node_from", "node_to", "flags"};
  uint32_t values[4] = {0};
  uint32_t present = 0;
  char* cursor = text;
  while (*cursor != 0) {
    char* separator = strchr(cursor, ' ');
    if (separator == NULL) return amdf_linux_error(EPROTO);
    *separator = 0;
    char* end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(separator + 1, &end, 10);
    if (errno || separator[1] < '0' || separator[1] > '9' ||
        end == separator + 1 || *end != '\n') {
      return amdf_linux_error(EPROTO);
    }
    for (uint32_t i = 0; i < 4; ++i) {
      if (strcmp(cursor, names[i]) != 0) continue;
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
  *out_backing_node = values[2];
  *out_enabled = values[0] == HSA_IOLINK_TYPE_PCIEXPRESS &&
                 (values[3] & (HSA_IOLINK_FLAGS_ENABLED |
                               HSA_IOLINK_FLAGS_NO_PEER_TO_PEER_DMA)) ==
                     HSA_IOLINK_FLAGS_ENABLED;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_kfd_read_memory_peers(
    int nodes, int node, uint32_t node_ordinal, amdf_allocator_t host_allocator,
    amdf_gpu_kfd_topology_t* topology) {
  const uint32_t link_count = topology->memory_peers.count;
  topology->memory_peers.count = 0;
  if (link_count == 0) return AMDF_STATUS_OK;
  amdf_status_t status = amdf_malloc(
      host_allocator, (uint64_t)link_count * sizeof(uint32_t),
      amdf_alignof(uint32_t), (void**)&topology->memory_peers.gpu_ids);
  for (uint32_t i = 0; amdf_status_is_ok(status) && i < link_count; ++i) {
    uint32_t backing_node = 0;
    bool enabled = false;
    status = amdf_gpu_kfd_read_peer_link(node, node_ordinal, i, &backing_node,
                                         &enabled);
    if (!amdf_status_is_ok(status) || !enabled) continue;
    char path[64];
    snprintf(path, sizeof(path), "%u/gpu_id", backing_node);
    uint32_t gpu_id = 0;
    status = amdf_gpu_kfd_read_number(nodes, path, &gpu_id);
    if (amdf_status_is_ok(status) && gpu_id != 0) {
      topology->memory_peers.gpu_ids[topology->memory_peers.count++] = gpu_id;
    }
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
  // APU VRAM requests may be redirected to GTT by KFD. System memory remains
  // available there without promising a physical placement the kernel changes.
  if (integrated || topology->vram.total_byte_length == 0) return 0;
  amdf_gpu_device_features_t features = AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY;
  if (topology->vram.visible_byte_length >= topology->vram.total_byte_length) {
    features |= AMDF_GPU_DEVICE_FEATURE_HOST_VISIBLE_LOCAL_MEMORY;
  }
  return features;
}

amdf_status_t amdf_gpu_kfd_topology_refine_memory(
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
       (device.virtual_address_alignment - 1)) != 0) {
    return amdf_linux_error(EPROTO);
  }
  topology->virtual_address.begin = device.virtual_address_offset;
  topology->virtual_address.end = device.virtual_address_max;
  topology->virtual_address.alignment = device.virtual_address_alignment;
  topology->memory_features = amdf_gpu_kfd_topology_memory_features(
      topology, (device.ids_flags & AMDGPU_IDS_FLAGS_FUSION) != 0);
  return AMDF_STATUS_OK;
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
  if (amdf_status_is_ok(status)) {
    nodes =
        openat(topology_directory, "nodes", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (nodes < 0) status = amdf_linux_error(errno);
  }
  DIR* directory = NULL;
  if (amdf_status_is_ok(status)) {
    directory = fdopendir(nodes);
    if (directory == NULL)
      status = amdf_linux_error(errno);
    else
      nodes = -1;
  }
  amdf_gpu_kfd_topology_t topology = {0};
  bool found = false;
  while (amdf_status_is_ok(status) && !found) {
    errno = 0;
    const struct dirent* entry = readdir(directory);
    if (entry == NULL) {
      if (errno) status = amdf_linux_error(errno);
      break;
    }
    if (entry->d_name[0] == 0 ||
        strspn(entry->d_name, "0123456789") != strlen(entry->d_name))
      continue;
    int node = openat(dirfd(directory), entry->d_name,
                      O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (node < 0) {
      status = amdf_linux_error(errno);
      break;
    }
    uint32_t gpu_id = 0;
    status = amdf_gpu_kfd_read_number(node, "gpu_id", &gpu_id);
    if (amdf_status_is_ok(status) && gpu_id != 0) {
      status = amdf_gpu_kfd_read_node(node, endpoint, &topology, &found);
      if (amdf_status_is_ok(status) && found) {
        topology.gpu_id = gpu_id;
        char* end = NULL;
        errno = 0;
        const unsigned long node_ordinal = strtoul(entry->d_name, &end, 10);
        if (errno || *end != 0 || node_ordinal > UINT32_MAX) {
          status = amdf_linux_error(EPROTO);
        } else {
          status = amdf_gpu_kfd_read_memory_peers(dirfd(directory), node,
                                                  (uint32_t)node_ordinal,
                                                  host_allocator, &topology);
        }
      }
    }
    const amdf_status_t close_status = amdf_linux_file_close(&node);
    if (!amdf_status_is_ok(close_status)) status = close_status;
  }
  if (amdf_status_is_ok(status) && found) {
    status = amdf_gpu_kfd_query_sdma(endpoint, &topology);
  }
  if (amdf_status_is_ok(status) && found) {
    status = amdf_gpu_kfd_read_memory(endpoint, &topology);
  }
  uint32_t final_generation = 0;
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_read_number(topology_directory, "generation_id",
                                      &final_generation);
    if (amdf_status_is_ok(status) && final_generation != generation) {
      status = amdf_linux_error(EAGAIN);
    }
  }
  if (directory != NULL && closedir(directory) != 0)
    status = amdf_linux_error(errno);
  const amdf_status_t nodes_status = amdf_linux_file_close(&nodes);
  if (!amdf_status_is_ok(nodes_status)) status = nodes_status;
  const amdf_status_t close_status = amdf_linux_file_close(&topology_directory);
  if (!amdf_status_is_ok(close_status)) status = close_status;
  if (amdf_status_is_ok(status) && !found) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (amdf_status_is_ok(status)) {
    *out_topology = topology;
  } else {
    amdf_gpu_kfd_topology_deinitialize(&topology, host_allocator);
  }
  return status;
}
