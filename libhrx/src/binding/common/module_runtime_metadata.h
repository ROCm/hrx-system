// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_MODULE_RUNTIME_METADATA_H_
#define IREE_EXPERIMENTAL_STREAMING_MODULE_RUNTIME_METADATA_H_

#include <stdbool.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_buffer_t iree_hal_streaming_buffer_t;
typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_module_t iree_hal_streaming_module_t;
typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;
typedef struct iree_hal_streaming_symbol_t iree_hal_streaming_symbol_t;

// Backend-native hidden runtime parameter slot.
typedef struct iree_hal_streaming_runtime_parameter_slot_t {
  // Byte offset in the backend-native launch argument storage.
  uint32_t offset;
  // Byte length of the runtime parameter storage.
  uint32_t length;
  // True when this slot was declared by executable metadata.
  bool present;
} iree_hal_streaming_runtime_parameter_slot_t;

// Runtime service slots declared by one function symbol.
typedef struct iree_hal_streaming_symbol_runtime_services_t {
  // Hidden buffered-printf service slot.
  iree_hal_streaming_runtime_parameter_slot_t printf_buffer;
  // Hidden hostcall service slot.
  iree_hal_streaming_runtime_parameter_slot_t hostcall_buffer;
  // Hidden device-runtime heap slot.
  iree_hal_streaming_runtime_parameter_slot_t heap_v1;
} iree_hal_streaming_symbol_runtime_services_t;

// Parsed printf format metadata retained at module scope.
typedef struct iree_hal_streaming_printf_format_t {
  // Lower 64 bits of the MD5 hash emitted in `amdhsa.printf`.
  uint64_t hash;
  // Format string borrowed from executable metadata storage.
  iree_string_view_t format;
} iree_hal_streaming_printf_format_t;

// Returns true when a dispatch must populate a backend-native runtime service.
bool iree_hal_streaming_symbol_uses_runtime_services(
    const iree_hal_streaming_symbol_t* symbol, bool enable_printf);

// Initializes hidden runtime service and executable-level runtime metadata.
iree_status_t iree_hal_streaming_module_initialize_runtime_metadata(
    iree_hal_streaming_module_t* module);

// Allocates and initializes a transient buffered-printf FIFO for one dispatch.
iree_status_t iree_hal_streaming_symbol_create_printf_buffer(
    iree_hal_streaming_symbol_t* symbol,
    iree_hal_streaming_buffer_t** out_buffer);

// Adds hidden runtime service pointers required by |symbol| to |config|.
// |runtime_parameters| is caller-owned dispatch storage that must remain live
// through command recording; |config| references it when it has patches. When
// |enable_printf| is true, returns a transient printf FIFO in
// |out_printf_buffer| that must be drained or released by the caller.
iree_status_t iree_hal_streaming_symbol_prepare_runtime_dispatch_config(
    iree_hal_streaming_symbol_t* symbol, iree_hal_streaming_context_t* context,
    bool enable_printf,
    iree_hal_dispatch_runtime_parameter_list_t* runtime_parameters,
    iree_hal_dispatch_config_t* config,
    iree_hal_streaming_buffer_t** out_printf_buffer);

// Enqueues a stream-ordered drain and release for a transient printf buffer.
// Takes ownership of |buffer| regardless of success.
iree_status_t iree_hal_streaming_symbol_queue_printf_drain(
    iree_hal_streaming_symbol_t* symbol, iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_MODULE_RUNTIME_METADATA_H_
