// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/platform/linux/endpoint.h"

#include <dirent.h>
#include <drm/drm.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/pci.h"
#include "libamdf/src/platform/linux/file.h"

// Sysfs attributes are bounded single records. A short read is their complete
// value; consuming the trailing newline is part of parsing each attribute.
static amdf_status_t amdf_linux_read_attribute(int directory, const char* name,
                                               char* value, size_t capacity) {
  int descriptor = openat(directory, name, O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return amdf_linux_error(errno);
  }
  ssize_t length;
  do {
    length = read(descriptor, value, capacity - 1);
  } while (length < 0 && errno == EINTR);
  amdf_status_t status = length < 0 ? amdf_linux_error(errno) : AMDF_STATUS_OK;
  if (length >= 0) {
    value[length] = 0;
    if ((size_t)length == capacity - 1) {
      status = amdf_linux_error(EOVERFLOW);
    }
  }
  const amdf_status_t close_status = amdf_linux_file_close(&descriptor);
  return amdf_status_is_ok(close_status) ? status : close_status;
}

static amdf_status_t amdf_linux_read_pci_attribute(int directory,
                                                   const char* name,
                                                   uint32_t* out_value) {
  char text[32];
  amdf_status_t status =
      amdf_linux_read_attribute(directory, name, text, sizeof(text));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  char* end = NULL;
  errno = 0;
  const unsigned long value = strtoul(text, &end, 0);
  if (errno || end == text || (*end != '\n' && *end != 0) ||
      (*end == '\n' && end[1] != 0) || value > UINT32_MAX) {
    return amdf_linux_error(EPROTO);
  }
  *out_value = (uint32_t)value;
  return AMDF_STATUS_OK;
}

static uint64_t amdf_linux_identity_hash(uint64_t hash, const void* bytes,
                                         size_t length) {
  const uint8_t* data = bytes;
  for (size_t i = 0; i < length; ++i) {
    hash = (hash ^ data[i]) * UINT64_C(1099511628211);
  }
  return hash;
}

// Returns NOT_FOUND for a class member outside the native AMD PCI providers.
// IO and malformed-attribute errors remain visible to the caller.
static amdf_status_t amdf_linux_query_endpoint(int directory,
                                               amdf_endpoint_info_t* info) {
  *info = (amdf_endpoint_info_t){
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(*info),
  };
  amdf_status_t status = amdf_linux_read_pci_attribute(
      directory, "device/vendor", &info->pci.vendor_id);
  if (status == amdf_linux_error(ENOENT) ||
      (amdf_status_is_ok(status) && !amdf_pci_is_amd(&info->pci))) {
    return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const char* attributes[] = {"device/device", "device/subsystem_vendor",
                              "device/subsystem_device", "device/revision"};
  uint32_t* fields[] = {&info->pci.device_id, &info->pci.subsystem_vendor_id,
                        &info->pci.subsystem_device_id, &info->pci.revision_id};
  for (size_t i = 0; amdf_status_is_ok(status) && i < 4; ++i) {
    status = amdf_linux_read_pci_attribute(directory, attributes[i], fields[i]);
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  char driver[256];
  const ssize_t driver_length =
      readlinkat(directory, "device/driver", driver, sizeof(driver) - 1);
  if (driver_length < 0) {
    return amdf_linux_error(errno);
  }
  if ((size_t)driver_length == sizeof(driver) - 1) {
    return amdf_linux_error(EOVERFLOW);
  }
  driver[driver_length] = 0;
  const char* driver_name = strrchr(driver, '/');
  driver_name = driver_name ? driver_name + 1 : driver;
  if (strcmp(driver_name, "amdxdna") == 0) {
    info->engine_kind = AMDF_ENGINE_KIND_XDNA;
  } else if (strcmp(driver_name, "amdgpu") == 0) {
    info->engine_kind = AMDF_ENGINE_KIND_GPU;
  } else {
    return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }

  char device_number[32];
  status = amdf_linux_read_attribute(directory, "dev", device_number,
                                     sizeof(device_number));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  unsigned int device_major, device_minor;
  char trailing;
  if (sscanf(device_number, "%u:%u%c", &device_major, &device_minor,
             &trailing) != 3 ||
      trailing != '\n') {
    return amdf_linux_error(EPROTO);
  }
  info->id.words[0] = ((uint64_t)device_major << 32) | device_minor;
  info->native_identity.type = AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE;
  info->native_identity.value.linux_device.major = device_major;
  info->native_identity.value.linux_device.minor = device_minor;
  char device_path[512];
  const ssize_t path_length =
      readlinkat(directory, "device", device_path, sizeof(device_path) - 1);
  if (path_length < 0) {
    return amdf_linux_error(errno);
  }
  if ((size_t)path_length == sizeof(device_path) - 1) {
    return amdf_linux_error(EOVERFLOW);
  }
  device_path[path_length] = 0;
  const char* pci_address = strrchr(device_path, '/');
  pci_address = pci_address ? pci_address + 1 : device_path;
  uint64_t hash = amdf_linux_identity_hash(UINT64_C(14695981039346656037),
                                           &info->pci, sizeof(info->pci));
  info->id.words[1] =
      amdf_linux_identity_hash(hash, pci_address, strlen(pci_address));
  info->type_flags = info->engine_kind == AMDF_ENGINE_KIND_XDNA
                         ? AMDF_ENDPOINT_TYPE_FLAG_COMPUTE_ONLY
                         : AMDF_ENDPOINT_TYPE_FLAG_RENDER_SUPPORTED;
  const int name_length =
      snprintf(info->name, sizeof(info->name), "AMD %s %04x:%04x:%02x (%s)",
               info->engine_kind == AMDF_ENGINE_KIND_XDNA ? "XDNA" : "GPU",
               info->pci.vendor_id, info->pci.device_id, info->pci.revision_id,
               pci_address);
  if (name_length < 0 || (size_t)name_length >= sizeof(info->name)) {
    return amdf_linux_error(EOVERFLOW);
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_platform_endpoint_enumerate(
    amdf_platform_instance_t* instance, uint32_t capacity,
    amdf_endpoint_summary_t* summaries, uint32_t* out_count) {
  if ((size_t)capacity > SIZE_MAX / sizeof(*summaries)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  amdf_endpoint_summary_t* staged_summaries = NULL;
  if (capacity != 0) {
    const amdf_status_t allocation_status = amdf_calloc_array(
        instance->host_allocator, (size_t)capacity, sizeof(*staged_summaries),
        amdf_alignof(amdf_endpoint_summary_t), (void**)&staged_summaries);
    if (!amdf_status_is_ok(allocation_status)) {
      return allocation_status;
    }
  }

  const char* classes[] = {"class/drm", "class/accel"};
  const char* prefixes[] = {"renderD", "accel"};
  uint32_t count = 0;
  amdf_status_t status = AMDF_STATUS_OK;
  for (size_t class_index = 0; amdf_status_is_ok(status) && class_index < 2;
       ++class_index) {
    int descriptor = openat(instance->sysfs_descriptor, classes[class_index],
                            O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (descriptor < 0) {
      // An absent class means the kernel has no devices in this class.
      if (errno != ENOENT) {
        status = amdf_linux_error(errno);
      }
      continue;
    }
    DIR* directory = fdopendir(descriptor);
    if (directory == NULL) {
      status = amdf_linux_error(errno);
      const amdf_status_t close_status = amdf_linux_file_close(&descriptor);
      if (!amdf_status_is_ok(close_status)) {
        status = close_status;
      }
      continue;
    }
    while (amdf_status_is_ok(status)) {
      errno = 0;
      const struct dirent* entry = readdir(directory);
      if (entry == NULL) {
        if (errno) {
          status = amdf_linux_error(errno);
        }
        break;
      }
      const size_t prefix_length = strlen(prefixes[class_index]);
      if (strncmp(entry->d_name, prefixes[class_index], prefix_length) != 0 ||
          entry->d_name[prefix_length] == 0 ||
          strspn(entry->d_name + prefix_length, "0123456789") !=
              strlen(entry->d_name + prefix_length)) {
        continue;
      }
      int node =
          openat(descriptor, entry->d_name, O_PATH | O_DIRECTORY | O_CLOEXEC);
      if (node < 0) {
        status = amdf_linux_error(errno);
        break;
      }
      amdf_endpoint_info_t info;
      status = amdf_linux_query_endpoint(node, &info);
      const amdf_status_t close_status = amdf_linux_file_close(&node);
      if (!amdf_status_is_ok(close_status)) {
        status = close_status;
      }
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND)) {
        status = AMDF_STATUS_OK;
        continue;
      }
      if (amdf_status_is_ok(status)) {
        if (count == UINT32_MAX) {
          status = amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
          break;
        }
        if (count < capacity) {
          staged_summaries[count] = (amdf_endpoint_summary_t){
              .id = info.id,
              .engine_kind = info.engine_kind,
              .type_flags = info.type_flags,
          };
          memcpy(staged_summaries[count].name, info.name, sizeof(info.name));
        }
        ++count;
      }
    }
    if (closedir(directory) != 0) {
      status = amdf_linux_error(errno);
    }
  }
  if (amdf_status_is_ok(status)) {
    if (capacity != 0) {
      const uint32_t copied_count = count < capacity ? count : capacity;
      memcpy(summaries, staged_summaries,
             (size_t)copied_count * sizeof(*summaries));
    }
    *out_count = count;
    if (capacity != 0 && capacity < count) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
    }
  }
  amdf_free(instance->host_allocator, staged_summaries);
  return status;
}

enum { AMDF_LINUX_DEVICE_PATH_CAPACITY = 64 };

static amdf_status_t amdf_linux_resolve_endpoint(
    amdf_platform_instance_t* instance, const amdf_endpoint_id_t* id,
    amdf_endpoint_info_t* info,
    char device_path[AMDF_LINUX_DEVICE_PATH_CAPACITY]) {
  char path[64];
  snprintf(path, sizeof(path), "dev/char/%u:%u", (uint32_t)(id->words[0] >> 32),
           (uint32_t)id->words[0]);
  int node = openat(instance->sysfs_descriptor, path,
                    O_PATH | O_DIRECTORY | O_CLOEXEC);
  if (node < 0) {
    return errno == ENOENT ? amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND)
                           : amdf_linux_error(errno);
  }
  amdf_status_t status = amdf_linux_query_endpoint(node, info);
  const amdf_status_t close_status = amdf_linux_file_close(&node);
  if (!amdf_status_is_ok(close_status)) {
    status = close_status;
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (!amdf_endpoint_id_is_equal(id, &info->id)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }
  char link[512];
  ssize_t length =
      readlinkat(instance->sysfs_descriptor, path, link, sizeof(link) - 1);
  if (length < 0) {
    return amdf_linux_error(errno);
  }
  if ((size_t)length == sizeof(link) - 1) {
    return amdf_linux_error(EOVERFLOW);
  }
  link[length] = 0;
  const char* name = strrchr(link, '/');
  if (name == NULL) {
    return amdf_linux_error(EPROTO);
  }
  const char* prefix =
      info->engine_kind == AMDF_ENGINE_KIND_XDNA ? "accel" : "renderD";
  ++name;
  const size_t prefix_length = strlen(prefix);
  if (strncmp(name, prefix, prefix_length) != 0 || name[prefix_length] == 0 ||
      strspn(name + prefix_length, "0123456789") !=
          strlen(name + prefix_length)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }
  const int path_length = snprintf(
      device_path, AMDF_LINUX_DEVICE_PATH_CAPACITY, "/dev/%s/%s",
      info->engine_kind == AMDF_ENGINE_KIND_XDNA ? "accel" : "dri", name);
  return (size_t)path_length < AMDF_LINUX_DEVICE_PATH_CAPACITY
             ? AMDF_STATUS_OK
             : amdf_linux_error(EOVERFLOW);
}

static amdf_status_t amdf_linux_qualify_device_file(
    int descriptor, const amdf_endpoint_info_t* info) {
  struct stat native_info;
  if (fstat(descriptor, &native_info) != 0) {
    return amdf_linux_error(errno);
  }
  if (!S_ISCHR(native_info.st_mode) ||
      major(native_info.st_rdev) != (uint32_t)(info->id.words[0] >> 32) ||
      minor(native_info.st_rdev) != (uint32_t)info->id.words[0]) {
    return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }
  char driver[32] = {0};
  struct drm_version version = {.name_len = sizeof(driver) - 1, .name = driver};
  if (ioctl(descriptor, DRM_IOCTL_VERSION, &version) != 0) {
    return amdf_linux_error(errno);
  }
  const char* expected_driver = info->engine_kind == AMDF_ENGINE_KIND_XDNA
                                    ? "amdxdna_accel_driver"
                                    : "amdgpu";
  if (version.name_len != strlen(expected_driver) ||
      memcmp(driver, expected_driver, version.name_len) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND);
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_linux_open_device_file(
    const amdf_endpoint_info_t* info, const char* path, int* out_descriptor) {
  int descriptor = open(path, O_RDWR | O_CLOEXEC);
  if (descriptor < 0) {
    return amdf_linux_error(errno);
  }
  amdf_status_t status = amdf_linux_qualify_device_file(descriptor, info);
  if (amdf_status_is_ok(status)) {
    *out_descriptor = descriptor;
  } else {
    const amdf_status_t close_status = amdf_linux_file_close(&descriptor);
    if (!amdf_status_is_ok(close_status)) {
      status = close_status;
    }
  }
  return status;
}

amdf_status_t amdf_platform_endpoint_open(
    amdf_platform_instance_t* instance, const amdf_endpoint_id_t* id,
    amdf_platform_endpoint_t** out_endpoint, amdf_endpoint_info_t* out_info) {
  amdf_platform_endpoint_t* endpoint = NULL;
  amdf_status_t status =
      amdf_calloc(instance->host_allocator, sizeof(*endpoint),
                  amdf_alignof(amdf_platform_endpoint_t), (void**)&endpoint);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  endpoint->instance = instance;
  char device_path[AMDF_LINUX_DEVICE_PATH_CAPACITY];
  status =
      amdf_linux_resolve_endpoint(instance, id, &endpoint->info, device_path);
  if (amdf_status_is_ok(status)) {
    *out_info = endpoint->info;
    *out_endpoint = endpoint;
  } else {
    amdf_free(instance->host_allocator, endpoint);
  }
  return status;
}

amdf_status_t amdf_linux_endpoint_open_file(
    const amdf_platform_endpoint_t* endpoint, int* out_descriptor) {
  amdf_endpoint_info_t info;
  char device_path[AMDF_LINUX_DEVICE_PATH_CAPACITY];
  const amdf_status_t status = amdf_linux_resolve_endpoint(
      endpoint->instance, &endpoint->info.id, &info, device_path);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  return amdf_linux_open_device_file(&info, device_path, out_descriptor);
}

amdf_queue_publication_modes_t
amdf_platform_endpoint_query_queue_publication_modes(
    const amdf_platform_endpoint_t* endpoint,
    amdf_queue_command_type_t command_type) {
  if (command_type == AMDF_QUEUE_COMMAND_TYPE_XDNA &&
      endpoint->info.engine_kind == AMDF_ENGINE_KIND_XDNA) {
    // Expected implemented transport; activation qualifies the native ABI.
    return AMDF_QUEUE_PUBLICATION_MODE_KERNEL;
  }
  return 0;
}

amdf_status_t amdf_platform_endpoint_close(amdf_platform_endpoint_t* endpoint) {
  amdf_free(endpoint->instance->host_allocator, endpoint);
  return AMDF_STATUS_OK;
}
