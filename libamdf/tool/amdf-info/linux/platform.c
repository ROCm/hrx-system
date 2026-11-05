// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/tool/amdf-info/platform.h"

#include <errno.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

const char* amdf_info_platform_name(void) { return "Linux"; }

bool amdf_info_platform_has_color(FILE* file) {
  return isatty(fileno(file)) != 0;
}

amdf_status_t amdf_info_platform_resolve_path(
    const char* path, amdf_endpoint_native_identity_t* out_identity) {
  struct stat metadata;
  if (stat(path, &metadata) != 0) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, (uint32_t)errno);
  }
  if (!S_ISCHR(metadata.st_mode)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  *out_identity = (amdf_endpoint_native_identity_t){
      .type = AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE,
      .value.linux_device =
          {
              .major = major(metadata.st_rdev),
              .minor = minor(metadata.st_rdev),
          },
  };
  return AMDF_STATUS_OK;
}
