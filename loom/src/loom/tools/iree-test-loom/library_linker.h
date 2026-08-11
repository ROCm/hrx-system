// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tool-private composition of explicit Loom library archives.

#ifndef LOOM_TOOLS_IREE_TEST_LOOM_LIBRARY_LINKER_H_
#define LOOM_TOOLS_IREE_TEST_LOOM_LIBRARY_LINKER_H_

#include "iree/base/api.h"
#include "loom/tooling/execution/session.h"

#ifdef __cplusplus
extern "C" {
#endif

// Links each explicit |library_paths| module whole into |run_module|.
//
// The primary run module is added first so its global declarations and private
// definitions retain input precedence. Library paths must name filesystem
// files; stdin is reserved for the primary input. On success |run_module| owns
// the linked replacement module while retaining its primary source metadata.
iree_status_t iree_test_loom_link_libraries(
    loom_run_session_t* session, loom_run_module_t* run_module,
    iree_string_view_list_t library_paths);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_TEST_LOOM_LIBRARY_LINKER_H_
