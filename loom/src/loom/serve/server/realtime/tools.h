// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_TOOLS_H_
#define LOOM_SERVE_SERVER_REALTIME_TOOLS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable declarations shared by session defaults and response snapshots.
// References belong to the serialized session owner, never to worker tasks.
// NULL denotes an empty catalogue.
typedef struct loom_serve_realtime_tools_t loom_serve_realtime_tools_t;

// Converts flat Realtime function declarations to native chat declarations and
// validates through the native schema owner. Function ranges are retained in
// declaration order for the flat session echo; no later reparsing is needed.
iree_status_t loom_serve_realtime_tools_parse(
    iree_string_view_t array, iree_allocator_t host_allocator,
    loom_serve_realtime_tools_t** out_tools);
void loom_serve_realtime_tools_retain(loom_serve_realtime_tools_t* tools);
void loom_serve_realtime_tools_release(loom_serve_realtime_tools_t* tools);
iree_string_view_t loom_serve_realtime_tools_native(
    const loom_serve_realtime_tools_t* tools);
iree_status_t loom_serve_realtime_tools_write(
    const loom_serve_realtime_tools_t* tools, iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_TOOLS_H_
