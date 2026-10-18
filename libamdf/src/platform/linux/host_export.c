// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/platform/host_export.h"

#include <fcntl.h>
#include <linux/udmabuf.h>
#include <sys/ioctl.h>

#include "libamdf/src/platform/linux/dma_buf.h"
#include "libamdf/src/platform/linux/file.h"

static amdf_status_t amdf_linux_host_export_query_limit(uint64_t* out_limit) {
  int descriptor = open("/sys/module/udmabuf/parameters/size_limit_mb",
                        O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return amdf_linux_error(errno);
  }
  // The module parameter is a signed int; only positive limits are usable.
  char text[12];
  ssize_t length;
  do {
    length = read(descriptor, text, sizeof(text));
  } while (length < 0 && errno == EINTR);
  amdf_status_t status = length < 0 ? amdf_linux_error(errno) : AMDF_STATUS_OK;
  const amdf_status_t close_status = amdf_linux_file_close(&descriptor);
  if (!amdf_status_is_ok(close_status)) {
    status = close_status;
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (length == 0 || length == sizeof(text)) {
    return amdf_linux_error(EPROTO);
  }
  if (text[length - 1] == '\n') {
    --length;
  }
  if (length == 0) {
    return amdf_linux_error(EPROTO);
  }
  uint64_t value = 0;
  for (ssize_t i = 0; i < length; ++i) {
    if (text[i] < '0' || text[i] > '9') {
      return amdf_linux_error(EPROTO);
    }
    value = value * 10 + (uint32_t)(text[i] - '0');
    if (value > INT32_MAX) {
      return amdf_linux_error(EPROTO);
    }
  }
  *out_limit = (uint64_t)value << 20;
  return value != 0 ? AMDF_STATUS_OK : amdf_linux_error(ENOSPC);
}

void amdf_platform_host_export_initialize(
    amdf_platform_host_export_t* service) {
  *service = (amdf_platform_host_export_t){
      .type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
      .native_handle = -1,
  };
  int descriptor = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
  if (descriptor < 0) {
    service->availability = amdf_linux_error(errno);
    return;
  }
  service->availability =
      amdf_linux_host_export_query_limit(&service->maximum_byte_length);
  if (amdf_status_is_ok(service->availability)) {
    service->native_handle = descriptor;
  } else {
    const amdf_status_t status = amdf_linux_file_close(&descriptor);
    if (!amdf_status_is_ok(status)) {
      service->availability = status;
    }
  }
}

amdf_status_t amdf_platform_host_export_deinitialize(
    amdf_platform_host_export_t* service) {
  int descriptor = (int)service->native_handle;
  service->native_handle = -1;
  return amdf_linux_file_close(&descriptor);
}

amdf_status_t amdf_platform_host_export_memory(
    const amdf_platform_host_export_t* service,
    const amdf_platform_host_backing_t* backing, uint64_t byte_length,
    amdf_external_memory_t* out_memory) {
  const struct udmabuf_create create = {
      .memfd = (uint32_t)backing->handle,
      .flags = UDMABUF_FLAGS_CLOEXEC,
      .size = byte_length,
  };
  int descriptor = ioctl((int)service->native_handle, UDMABUF_CREATE, &create);
  if (descriptor < 0) {
    return amdf_linux_error(errno);
  }
  amdf_linux_dma_buf_info_t info;
  amdf_status_t status = amdf_linux_dma_buf_query(descriptor, &info);
  if (amdf_status_is_ok(status)) {
    *out_memory = (amdf_external_memory_t){
        .type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
        .payload.file_descriptor = descriptor,
        .byte_length = byte_length,
        .physical_backing_id = info.physical_backing_id,
        .release = amdf_linux_dma_buf_release,
    };
  } else {
    const amdf_status_t close_status = amdf_linux_file_close(&descriptor);
    if (!amdf_status_is_ok(close_status)) {
      status = close_status;
    }
  }
  return status;
}
