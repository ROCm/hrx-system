// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_SOURCES_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_SOURCES_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed program inputs, independent of modality and registration policy.
// The owner retains every view through asynchronous model preparation. Authored
// locators remain separate from the local base used by an I/O consumer.
typedef struct loom_serve_sources_t {
  // Local model-document directory; empty requires absolute input paths.
  iree_string_view_t base_directory;
  // Cold VM program declaring stages and model metadata.
  iree_string_view_t prepare;
  // Warm VM program orchestrating requests and stage submissions.
  iree_string_view_t control;
  // Number of native command/kernel/configuration providers.
  iree_host_size_t count;
  // Authored provider paths; symbols establish dependencies, not list order.
  const iree_string_view_t* values;
} loom_serve_sources_t;

// Reports whether a locator is absolute in the local filesystem namespace.
bool loom_serve_path_is_absolute(iree_string_view_t path);

// Binds an authored locator to a local I/O base without changing its owner.
// Absolute locators ignore base; relative locators require a nonempty base.
// Windows drive-relative and root-relative paths reject instead of inheriting
// process drive state. Does not open the file or resolve symlinks. The caller
// frees the result.
iree_status_t loom_serve_path_resolve(iree_string_view_t base,
                                      iree_string_view_t path,
                                      iree_allocator_t allocator,
                                      char** out_path);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_SOURCES_H_
