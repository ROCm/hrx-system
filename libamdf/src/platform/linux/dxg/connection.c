// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/linux/dxg/connection.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include "libamdf/src/platform/linux/file.h"

_Static_assert(sizeof(unsigned long) == sizeof(uint64_t),
               "DXG ioctl requests require Linux x86-64 unsigned long");

static uint32_t amdf_linux_dxg_errno_to_ntstatus(int error) {
  switch (error) {
    case EEXIST:
      return 0xC0000035u;  // STATUS_OBJECT_NAME_COLLISION
    case ENOMEM:
      return 0xC0000017u;  // STATUS_NO_MEMORY
    case EINVAL:
      return 0xC000000Du;  // STATUS_INVALID_PARAMETER
    case ENOENT:
      return 0xC0000033u;  // STATUS_OBJECT_NAME_INVALID
    case EAGAIN:
      return 0x00000102u;  // STATUS_TIMEOUT
    case EOVERFLOW:
      return 0xC0000023u;  // STATUS_BUFFER_TOO_SMALL
    case ENODEV:
      return 0xC00002B6u;  // STATUS_DEVICE_REMOVED
    case EACCES:
      return 0xC0000022u;  // STATUS_ACCESS_DENIED
    case ENOSYS:
    case EPERM:
      return 0xC00000BBu;  // STATUS_NOT_SUPPORTED
    case EOPNOTSUPP:
      return 0xC000001Du;  // STATUS_ILLEGAL_INSTRUCTION
    case EBADF:
      return 0xC0000008u;  // STATUS_INVALID_HANDLE
    case EINPROGRESS:
      return 0xC01E0102u;  // STATUS_GRAPHICS_ALLOCATION_BUSY
    case EPROTOTYPE:
      return 0xC0000024u;  // STATUS_OBJECT_TYPE_MISMATCH
    default:
      return 0xC0000001u;  // STATUS_UNSUCCESSFUL
  }
}

amdf_status_t amdf_linux_dxg_connection_initialize(
    amdf_linux_dxg_connection_t* out_connection) {
  const int descriptor = open("/dev/dxg", O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return amdf_linux_error(errno);
  }
  *out_connection = (amdf_linux_dxg_connection_t){
      .descriptor = descriptor,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_linux_dxg_connection_deinitialize(
    amdf_linux_dxg_connection_t* connection) {
  return amdf_linux_file_close(&connection->descriptor);
}

uint32_t amdf_linux_dxg_connection_invoke(void* context, uint64_t request,
                                          void* arguments) {
  const amdf_linux_dxg_connection_t* connection = context;
  const int result =
      ioctl(connection->descriptor, (unsigned long)request, arguments);
  return result == -1 ? amdf_linux_dxg_errno_to_ntstatus(errno)
                      : (uint32_t)result;
}
