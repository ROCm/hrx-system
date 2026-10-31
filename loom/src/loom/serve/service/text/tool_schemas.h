// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_TOOL_SCHEMAS_H_
#define LOOM_SERVE_SERVICE_TEXT_TOOL_SCHEMAS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Protocol resource bound for declared or generated function tools. Storage is
// allocated only for the declarations present in a request.
#define LOOM_SERVE_TEXT_CHAT_MAX_TOOL_COUNT 4096

// Indexed representation hints for a function. The model-selected codec uses
// parameter types to decode XML text, not to validate executable arguments.
typedef struct loom_serve_text_chat_tool_t {
  // Decoded function name, borrowing the request unless it contains escapes.
  iree_string_view_t name;
  // Opaque parameter schema object, or empty when absent or null.
  iree_string_view_t parameters;
  // Owned name bytes only when JSON decoding required storage.
  char* name_storage;
} loom_serve_text_chat_tool_t;

// Function-envelope index owning descriptors and borrowing caller-owned JSON.
// Sorting descriptors for lookup never changes the original declaration order
// observed by source rendering.
typedef struct loom_serve_text_tool_schemas_t {
  // Allocator owning the descriptor array.
  iree_allocator_t host_allocator;
  // Number of populated descriptors.
  iree_host_size_t count;
  // Allocated descriptor capacity.
  iree_host_size_t capacity;
  // Function envelopes sorted by decoded function name.
  loom_serve_text_chat_tool_t* values;
} loom_serve_text_tool_schemas_t;

// Indexes function representation hints from the caller's validated UTF-8 JSON.
// Entries without a function name supply no hints; other catalog metadata is
// not interpreted. The JSON bytes outlive this table. Failure frees partial
// storage and leaves out_schemas empty and safe to deinitialize.
iree_status_t loom_serve_text_tool_schemas_initialize(
    iree_string_view_t declarations, iree_allocator_t host_allocator,
    loom_serve_text_tool_schemas_t* out_schemas);
void loom_serve_text_tool_schemas_deinitialize(
    loom_serve_text_tool_schemas_t* schemas);

// Returns the unique function's hints or NULL for absent or repeated names.
const loom_serve_text_chat_tool_t* loom_serve_text_tool_schemas_find(
    const loom_serve_text_tool_schemas_t* schemas, iree_string_view_t name);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_TOOL_SCHEMAS_H_
