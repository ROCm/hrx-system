// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_FABRIC_FILE_TRANSFER_H_
#define IREE_HAL_DRIVER_AMD_FABRIC_FILE_TRANSFER_H_

#include "amdf/amdf.h"
#include "iree/async/file.h"
#include "iree/hal/api.h"
#include "iree/hal/driver/amd/fabric/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

// One contiguous physical source of a file operation. Source identity and
// coordinates remain available when native copies are required. Host views
// and optional registrations are prepared by the backing owner and borrowed.
typedef struct iree_hal_amd_fabric_file_source_t {
  // Borrowed native API, dominated by the application's device ownership.
  const amdf_api_t* api;
  // Borrowed physical source; installed aliases do not retain its lifetime.
  const iree_hal_amd_fabric_memory_t* storage;
  // Borrowed retained registration, or NULL for raw I/O and engine copies.
  iree_async_region_t* region;
  // Exact extent of this source in the operation. A NULL data pointer requires
  // an engine copy; otherwise native file I/O can access the bytes directly.
  iree_byte_span_t contents;
  // Logical source-relative position, independent of native allocation padding.
  uint64_t source_offset;
  // Exact native consumer of this source, established during resolution.
  uint32_t access_ordinal;
  // Qualified effects at the file engine: host cache actions for direct views,
  // or copy-engine cache actions for ranges requiring native device copies.
  iree_hal_memory_effects_t effects;
} iree_hal_amd_fabric_file_source_t;

// Qualifies a captured buffer binding once after allocation dependencies.
// File/buffer ranges and access are already validated. The returned source
// borrows backing and any prepared registration; no resource is retained.
// Native visibility depends only on this buffer, queue, and access direction,
// so unrelated file requests can reuse it. Before using a registered span,
// the caller checks that its region belongs to the file's exact proactor.
// A NULL contents.data requires native copies or an admitted host binding.
iree_status_t iree_hal_amd_fabric_file_source_initialize(
    const amdf_api_t* api, const iree_hal_queue_family_t* family,
    uint32_t native_family_ordinal, iree_hal_memory_access_t file_access,
    iree_hal_buffer_binding_t binding,
    iree_hal_amd_fabric_file_source_t* out_source,
    iree_hal_memory_effects_t* out_before,
    iree_hal_memory_effects_t* out_after);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_FABRIC_FILE_TRANSFER_H_
