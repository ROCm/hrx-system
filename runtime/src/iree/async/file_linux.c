// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AT_EMPTY_PATH requires the GNU fcntl declarations on glibc.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif  // _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <linux/btrfs.h>
#include <linux/stat.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "iree/async/file_platform.h"

iree_status_t iree_async_file_query_direct_requirements(
    iree_async_primitive_t primitive,
    iree_async_file_requirements_t* out_requirements) {
  if (primitive.type != IREE_ASYNC_PRIMITIVE_TYPE_FD) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "file requirements query needs a file descriptor");
  }
  const int fd = primitive.value.fd;
  struct statx attributes = {0};
  int result = syscall(SYS_statx, fd, "", AT_EMPTY_PATH,
                       STATX_DIOALIGN | STATX_DIO_READ_ALIGN, &attributes);
  uint32_t memory_alignment = 0;
  uint32_t write_alignment = 0;
  uint32_t read_alignment = 0;
  if (result == 0 && (attributes.stx_mask & STATX_DIOALIGN)) {
    memory_alignment = attributes.stx_dio_mem_align;
    write_alignment = attributes.stx_dio_offset_align;
    read_alignment = (attributes.stx_mask & STATX_DIO_READ_ALIGN)
                         ? attributes.stx_dio_read_offset_align
                         : write_alignment;
  } else {
    if (result < 0 && errno != ENOSYS && errno != EINVAL &&
        errno != EOPNOTSUPP) {
      return iree_make_status(iree_status_code_from_errno(errno),
                              "statx direct I/O requirements query failed");
    }
    // Btrfs checks buffer and extent alignment against its filesystem sector
    // size, including kernels that do not expose STATX_DIOALIGN. This query
    // neither changes inode policy nor promises the absence of buffered paths.
    struct btrfs_ioctl_fs_info_args filesystem = {0};
    if (ioctl(fd, BTRFS_IOC_FS_INFO, &filesystem) != 0) {
      int error = errno;
      if (error == ENOTTY || error == EOPNOTSUPP || error == EINVAL) {
        return iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "filesystem does not report direct I/O alignment requirements");
      }
      return iree_make_status(iree_status_code_from_errno(error),
                              "Btrfs direct I/O requirements query failed");
    }
    memory_alignment = filesystem.sectorsize;
    write_alignment = filesystem.sectorsize;
    read_alignment = filesystem.sectorsize;
  }
  if (!iree_host_size_is_power_of_two(memory_alignment) ||
      !iree_host_size_is_power_of_two(write_alignment) ||
      !iree_host_size_is_power_of_two(read_alignment)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "filesystem has no usable direct I/O alignment");
  }
  // Fixed io_uring buffers reach direct I/O as page-bounded BVEC segments.
  // Aligning the first byte to the transfer unit keeps every segment boundary
  // legal instead of admitting a raw pointer geometry that READ_FIXED or
  // WRITE_FIXED may reject after registration.
  *out_requirements = (iree_async_file_requirements_t){
      .read = {iree_max(memory_alignment, read_alignment), read_alignment,
               read_alignment},
      .write = {iree_max(memory_alignment, write_alignment), write_alignment,
                write_alignment},
  };
  return iree_ok_status();
}
