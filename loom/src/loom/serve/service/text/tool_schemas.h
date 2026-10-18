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

// Indexed function-tool schema borrowing the request body. Names are literal
// ASCII identifiers. The optional XML codec uses property types to recover
// generated JSON values; the agent owns execution and full schema validation.
typedef struct loom_serve_text_chat_tool_t {
  // Literal function name, without JSON quotes.
  iree_string_view_t name;
  // JSON object mapping parameter names to schemas.
  iree_string_view_t properties;
  // JSON array of required parameter names, or empty when absent.
  iree_string_view_t required;
} loom_serve_text_chat_tool_t;

// Validated schema index owning descriptors and borrowing caller-owned JSON.
// Sorting descriptors for lookup never changes the original declaration order
// observed by source rendering.
typedef struct loom_serve_text_tool_schemas_t {
  // Allocator owning the descriptor array.
  iree_allocator_t host_allocator;
  // Number of populated descriptors.
  iree_host_size_t count;
  // Allocated descriptor capacity.
  iree_host_size_t capacity;
  // Validated descriptors sorted by literal function name.
  loom_serve_text_chat_tool_t* values;
} loom_serve_text_tool_schemas_t;

// Parses native function declarations from the caller's validated UTF-8 input.
// The JSON bytes outlive this table. Failure frees partial storage and leaves
// out_schemas empty and safe to deinitialize.
iree_status_t loom_serve_text_tool_schemas_initialize(
    iree_string_view_t declarations, iree_allocator_t host_allocator,
    loom_serve_text_tool_schemas_t* out_schemas);
void loom_serve_text_tool_schemas_deinitialize(
    loom_serve_text_tool_schemas_t* schemas);

// Returns the declared function or NULL when the name was not declared.
const loom_serve_text_chat_tool_t* loom_serve_text_tool_schemas_find(
    const loom_serve_text_tool_schemas_t* schemas, iree_string_view_t name);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_TOOL_SCHEMAS_H_
