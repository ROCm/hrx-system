// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/tool/amdf-info/platform.h"

#include <io.h>
#include <windows.h>

const char* amdf_info_platform_name(void) { return "Windows"; }

bool amdf_info_platform_has_color(FILE* file) {
  const intptr_t handle = _get_osfhandle(_fileno(file));
  DWORD mode = 0;
  return handle != -1 && GetConsoleMode((HANDLE)handle, &mode) &&
         (mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
}

amdf_status_t amdf_info_platform_resolve_path(
    const char* path, amdf_endpoint_native_identity_t* out_identity) {
  (void)path;
  (void)out_identity;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}
