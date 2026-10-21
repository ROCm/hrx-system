// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLING_TARGET_VM_BUFFER_H_
#define LOOM_TOOLING_TARGET_VM_BUFFER_H_

#include "iree/hal/buffer.h"
#include "iree/vm/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Wraps a persistent HAL mapping without copying bytes. The returned VM buffer
// retains the HAL buffer, including through VM subspans that escape a call.
// Callers order device accesses before invalidation and flush completed VM
// writes before returning ownership to a device.
iree_status_t loom_vm_buffer_import(iree_hal_buffer_t* source,
                                    iree_device_size_t byte_offset,
                                    iree_device_size_t byte_length,
                                    iree_allocator_t host_allocator,
                                    iree_vm_buffer_t** out_buffer);

// Invalidates readable noncoherent imported bytes before VM execution. VM-owned
// storage and coherent imports require no cache operation.
iree_status_t loom_vm_buffer_invalidate(iree_vm_buffer_t* buffer);

// Publishes writable noncoherent imported bytes after synchronous VM execution,
// including partial writes when execution fails. Other storage needs no action.
iree_status_t loom_vm_buffer_flush(iree_vm_buffer_t* buffer);

// Returns a retained HAL view of the VM buffer's exact range. Imported storage
// preserves the original HAL allocation and cache operations. VM-owned storage
// becomes a coherent host view retaining the VM buffer. No bytes are copied.
iree_status_t loom_vm_buffer_export(iree_vm_buffer_t* source,
                                    iree_allocator_t host_allocator,
                                    iree_hal_buffer_t** out_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TARGET_VM_BUFFER_H_
