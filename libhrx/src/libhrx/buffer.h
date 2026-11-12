// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_LIBHRX_BUFFER_H_
#define LIBHRX_SRC_LIBHRX_BUFFER_H_

#include "hrx_runtime.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Private representation of an HRX buffer allocation.
typedef struct hrx_buffer_s {
  // References held by callers of the public buffer handle.
  iree_atomic_ref_count_t ref_count;

  // HAL buffer owning the underlying allocation.
  iree_hal_buffer_t* hal_buffer;

  // Retained sealed group defining the allocation's memory scopes.
  hrx_device_group_t group;

  // Memory properties selected for the allocation.
  hrx_memory_type_t mem_type;

  // User-visible allocation length in bytes.
  size_t size;

  // Optional in-tree owner's state, valid until |release_callback| returns.
  void* release_user_data;

  // Synchronous final-release hook after native backing is released. Receives
  // |release_user_data| and the immutable logical |size|; never queued as work.
  void (*release_callback)(void* user_data, size_t byte_length);

  // Active scoped mapping, valid only while |is_mapped| is true.
  iree_hal_buffer_mapping_t mapping;

  // True when |mapping| currently owns an active mapping.
  bool is_mapped;

  // Optional caller-supplied whole-buffer device address, independent of
  // mapping.
  void* device_ptr;
} hrx_buffer_s;

// Creates an HRX buffer wrapping |hal_buffer| for in-tree interop. The buffer
// retains both |hal_buffer| and |group|; the caller owns the returned handle.
// Allocation pools remain caller-owned and must outlive the buffer.
// |hal_buffer| may be NULL for host-only allocations.
// |device_ptr| is an optional whole-buffer device address whose validity the
// caller guarantees for the lifetime of the wrapper. It is never a scoped map.
iree_status_t hrx_buffer_create_from_hal(iree_hal_buffer_t* hal_buffer,
                                         hrx_device_group_t group,
                                         hrx_memory_type_t mem_type,
                                         size_t size, void* device_ptr,
                                         hrx_buffer_t* out_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_LIBHRX_BUFFER_H_
