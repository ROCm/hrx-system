// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/sources.h"

#include "iree/base/internal/path.h"

bool loom_serve_path_is_absolute(iree_string_view_t path) {
#if defined(IREE_PLATFORM_WINDOWS)
  return (path.size >= 2 && ((path.data[0] == '\\' && path.data[1] == '\\') ||
                             (path.data[0] == '/' && path.data[1] == '/'))) ||
         (path.size >= 3 && path.data[1] == ':' &&
          ((path.data[0] >= 'A' && path.data[0] <= 'Z') ||
           (path.data[0] >= 'a' && path.data[0] <= 'z')) &&
          (path.data[2] == '/' || path.data[2] == '\\'));
#else
  return path.size && path.data[0] == '/';
#endif
}

iree_status_t loom_serve_path_resolve(iree_string_view_t base,
                                      iree_string_view_t path,
                                      iree_allocator_t allocator,
                                      char** out_path) {
  *out_path = NULL;
  if (loom_serve_path_is_absolute(path)) {
    return iree_file_path_join(iree_string_view_empty(), path, allocator,
                               out_path);
  }
#if defined(IREE_PLATFORM_WINDOWS)
  // A root-relative or drive-relative name depends on process drive state,
  // not the model document. It is neither a portable relative locator nor a
  // fully qualified local path.
  if ((path.size && (path.data[0] == '/' || path.data[0] == '\\')) ||
      (path.size >= 2 && path.data[1] == ':')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model paths must be relative or fully qualified");
  }
#endif
  if (iree_string_view_is_empty(base)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "relative model path '%.*s' requires a base directory", (int)path.size,
        path.data);
  }
  return iree_file_path_join(base, path, allocator, out_path);
}
