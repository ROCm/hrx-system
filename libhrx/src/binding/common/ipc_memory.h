// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_STREAMING_IPC_MEMORY_H_
#define IREE_HAL_STREAMING_IPC_MEMORY_H_

#include "common/memory.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define IREE_HAL_STREAMING_IPC_MEMORY_TOKEN_SIZE 32

typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_device_registry_t
    iree_hal_streaming_device_registry_t;
typedef struct iree_hal_streaming_ipc_memory_import_t
    iree_hal_streaming_ipc_memory_import_t;

// Reserves one IPC import publication against concurrent context retirement.
// The caller must hold a context reference until the matching end call.
// Synchronization: thread-safe internal locking.
iree_status_t iree_hal_streaming_context_try_begin_ipc_import(
    iree_hal_streaming_context_t* context);

// Releases one successful IPC import reservation.
// Synchronization: thread-safe internal locking.
void iree_hal_streaming_context_end_ipc_import(
    iree_hal_streaming_context_t* context);

// Permanently closes IPC-import admission and waits for every import admitted
// before the gate closed to complete or roll back. One retirement transaction
// owns the gate until it commits or aborts.
// Synchronization: thread-safe internal locking; may block.
void iree_hal_streaming_context_begin_ipc_import_retirement(
    iree_hal_streaming_context_t* context);

// Commits the current IPC-import retirement transaction, leaving admission
// permanently closed for this materialized context.
// Synchronization: thread-safe internal locking.
void iree_hal_streaming_context_commit_ipc_import_retirement(
    iree_hal_streaming_context_t* context);

// Reopens IPC-import admission when the teardown that began retirement aborts
// and leaves the context live, but only if this transaction closed admission.
// No import reservation may remain active.
// Synchronization: thread-safe internal locking.
void iree_hal_streaming_context_cancel_ipc_import_retirement(
    iree_hal_streaming_context_t* context);

// Backend-neutral identity and placement metadata for one exported allocation
// view. The fixed token width matches the interprocess handle contract used by
// the streaming API bindings.
typedef struct iree_hal_streaming_ipc_memory_descriptor_t {
  // Opaque backend token naming the complete shareable allocation.
  uint8_t token[IREE_HAL_STREAMING_IPC_MEMORY_TOKEN_SIZE];
  // Backend-reported shareable extent required by attach. Backend-private
  // physical padding is not included.
  uint64_t allocation_size;
  // Offset of the exported view from the shared allocation base.
  uint64_t byte_offset;
  // Process that exported the allocation.
  int64_t exporter_process_id;
  // Exporter's device ordinal in the shared visible-device namespace.
  uint64_t exporter_device_ordinal;
} iree_hal_streaming_ipc_memory_descriptor_t;

// Exports |buffer| synchronously into |out_descriptor|. The callback neither
// retains the context nor takes ownership of the buffer.
typedef iree_status_t (*iree_hal_streaming_ipc_memory_export_fn_t)(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* buffer,
    iree_hal_streaming_ipc_memory_descriptor_t* out_descriptor);

// Creates the one HAL buffer that owns the process-wide backend attachment.
// On success the caller owns |out_buffer|. On failure no attachment or buffer
// ownership may remain. The registry invokes this outside its mutex.
typedef iree_status_t (*iree_hal_streaming_ipc_memory_attach_fn_t)(
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_ipc_memory_descriptor_t* descriptor,
    iree_device_size_t view_size, iree_hal_buffer_t** out_buffer);

// Creates a context-local HAL buffer alias for a registry-owned attached IPC
// buffer. On success the caller owns the returned alias, which must not own or
// detach the shared attachment. The registry invokes this outside its mutex.
typedef iree_status_t (*iree_hal_streaming_ipc_memory_alias_fn_t)(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* attached_buffer,
    iree_hal_buffer_t** out_alias_buffer);

// Process-wide registry ensuring one driver attach per imported token/offset.
typedef struct iree_hal_streaming_ipc_memory_registry_t {
  // Guards the complete import list, entry states, and duplicate open counts.
  iree_slim_mutex_t mutex;
  // Posted whenever an entry finishes an attach or detach transition.
  iree_notification_t changed;
  // Head of the live imported-allocation list.
  iree_hal_streaming_ipc_memory_import_t* imports;
  // Host allocator used for import entries.
  iree_allocator_t host_allocator;
} iree_hal_streaming_ipc_memory_registry_t;

// Initializes an empty process-wide IPC memory registry.
void iree_hal_streaming_ipc_memory_registry_initialize(
    iree_hal_streaming_ipc_memory_registry_t* registry,
    iree_allocator_t host_allocator);

// Releases every mapping still owned by |registry| during global shutdown.
void iree_hal_streaming_ipc_memory_registry_deinitialize(
    iree_hal_streaming_ipc_memory_registry_t* registry);

// Exports an eligible live allocation and returns its backend descriptor and
// owning device ordinal.
iree_status_t iree_hal_streaming_ipc_memory_export(
    iree_hal_streaming_device_registry_t* device_registry,
    iree_hal_streaming_context_t* preferred_context, uint64_t device_ptr,
    iree_hal_streaming_ipc_memory_export_fn_t export_fn,
    iree_hal_streaming_ipc_memory_descriptor_t* out_descriptor,
    iree_host_size_t* out_device_ordinal);

// Attaches |descriptor| or reuses the one process-wide attachment for its token
// and byte-offset key. All successful opens increment one process-wide count
// because close identifies the cached object only by its shared address. A
// different device receives a persistent context-local alias from |alias_fn|;
// all live handles from one exporter process/device use one context per
// importing device. Backend attach, detach, and alias creation execute without
// the registry mutex held.
iree_status_t iree_hal_streaming_ipc_memory_import(
    iree_hal_streaming_ipc_memory_registry_t* registry,
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_ipc_memory_descriptor_t* descriptor,
    iree_device_size_t view_size,
    iree_hal_streaming_ipc_memory_attach_fn_t attach_fn,
    iree_hal_streaming_ipc_memory_alias_fn_t alias_fn, void** out_device_ptr);

// Closes one process-wide open reference identified by an exact pointer from
// iree_hal_streaming_ipc_memory_import. The reference attributed to |context|
// is consumed when one remains, with another live binding used as a fallback
// for HIP's cross-device cached-address close behavior. Each close waits for
// already-submitted work in every retained binding context, and the final close
// removes all bindings. The caller must exclude concurrent use or submission
// through the closing pointer.
iree_status_t iree_hal_streaming_ipc_memory_close(
    iree_hal_streaming_ipc_memory_registry_t* registry,
    iree_hal_streaming_context_t* context, void* device_ptr);

// Permanently closes IPC-import admission for |context|, waits for imports
// admitted before that gate to finish, and revokes every context-owned open
// reference and wrapper. Shared attachments and aliases owned by other
// contexts remain live. A final context-owned reference detaches the
// process-wide attachment. A failed drain reopens admission because the caller
// must leave the context live, and leaves every affected registry entry and
// mapping unchanged. Each affected import serializes with concurrent open,
// close, and context release. The caller must separately exclude ordinary
// pointer use or submission through the context being retired.
iree_status_t iree_hal_streaming_ipc_memory_release_context(
    iree_hal_streaming_ipc_memory_registry_t* registry,
    iree_hal_streaming_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_STREAMING_IPC_MEMORY_H_
