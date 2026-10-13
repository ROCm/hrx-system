// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_EXECUTABLE_INTERNAL_H_
#define IREE_HAL_DRIVERS_AMDXDNA_EXECUTABLE_INTERNAL_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/amdxdna/ctrlcode_elf.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer_planning.h"
#include "iree/hal/drivers/amdxdna/executable.h"
#include "iree/hal/drivers/amdxdna/native.h"

typedef struct iree_hal_amdxdna_context_cache_lease_t
    iree_hal_amdxdna_context_cache_lease_t;

typedef struct iree_hal_amdxdna_u8_list_t {
  uint8_t* data;
  iree_host_size_t count;
} iree_hal_amdxdna_u8_list_t;

typedef struct iree_hal_amdxdna_u32_list_t {
  uint32_t* data;
  iree_host_size_t count;
} iree_hal_amdxdna_u32_list_t;

typedef struct iree_hal_amdxdna_kernel_params_t {
  // Raw PDI context image from amdxdna-pdi-fb or extracted from an XADX
  // xclbin's AIE_PARTITION section for native drivers that consume PDI.
  // Empty for amdxdna-elf-fb (AIE4 START_DPU creates a context with no image).
  iree_hal_amdxdna_u8_list_t pdi;
  // AXLF/xclbin context wrapper from amdxdna-xclbin-fb for native
  // drivers that consume an xclbin-shaped context blob. Empty for AELF.
  iree_hal_amdxdna_u8_list_t xclbin;
  iree_hal_amdxdna_u32_list_t* asm_inst_runlist;
  iree_host_size_t asm_inst_runlist_count;
  iree_hal_amdxdna_u32_list_t* reconf_data_runlist;
  iree_host_size_t reconf_data_runlist_count;
  // Host patch table parallel to `asm_inst_runlist`: each inner list is a flat
  // list of (offset, arg_idx, arg_plus) triples for the corresponding control
  // code, applied by the ERT_CMD_CHAIN path.
  iree_hal_amdxdna_u32_list_t* patch_runlist;
  iree_host_size_t patch_runlist_count;
  // Standalone-PDI relocations parallel to `asm_inst_runlist`.
  iree_hal_amdxdna_pdi_relocation_list_t* pdi_relocation_runlist;
  iree_host_size_t pdi_relocation_runlist_count;
  // Internal control-parameter-buffer relocations parallel to the runlist.
  iree_hal_amdxdna_control_parameter_relocation_list_t*
      control_parameter_relocation_runlist;
  iree_host_size_t control_parameter_relocation_runlist_count;
  iree_hal_amdxdna_write32_constant_patch_list_t* constant_patch_runlist;
  iree_host_size_t constant_patch_runlist_count;
  // Logical scalar parameters passed in the dense dispatch constants block.
  // native_abi_offset maps each value to its four-byte firmware state-table
  // slot in the internally staged control parameter buffer.
  iree_hal_executable_function_parameter_t* parameters;
  iree_host_size_t parameter_count;
  uint32_t constant_byte_length;
  iree_string_view_t kernel_name;
  uint32_t n_reconfigure_runs;
  uint32_t n_pdi_loads;
  // Legacy Windows MCDM requires stable per-entry-point native context/CU
  // bindings for self-contained dispatches. The lease pins the cache entry
  // against LRU eviction while keeping ownership accounted by the device cache.
  // Written under the owning executable's context_mutex when native caps
  // request this compatibility path.
  iree_hal_amdxdna_context_cache_lease_t* cached_context_lease;
  // AIE4 ELF `.note.xrt.configuration` column count (XRT partition_size). 0
  // means the note was absent; native create then uses a 1-column empty ctx.
  uint32_t partition_cols;
  // Paged ELF occupied-column slices of asm_inst_runlist[0]. Owned here.
  // Count 0 or 1 submits one direct START_DPU packet; count > 1 is XRT's
  // per-column ert_dpu_data chain.
  iree_hal_amdxdna_ctrlcode_dpu_slice_t* dpu_slices;
  iree_host_size_t dpu_slice_count;
  iree_hal_amdxdna_native_c_cu_index_t cached_cu_index;
  bool cached_context_valid;
  IREE_TRACE(iree_string_view_t source_filename;)
  IREE_TRACE(uint32_t source_line;)
} iree_hal_amdxdna_kernel_params_t;

typedef struct iree_hal_amdxdna_executable {
  // Abstract resource used for injecting reference counting and vtable; must be
  // at offset 0.
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_amdxdna_native_device_t* native_device;
  // HAL device that loaded this executable. NULL for executables created
  // directly in tests. Destruction uses it to end buffer reuse and drop
  // hardware contexts that no executable still leases.
  struct iree_hal_amdxdna_device* hal_device;
  // Process-unique identity used by native command caches. Unlike the object
  // address this is never reused after destruction, so cached immutable
  // entry-point descriptors cannot alias a later executable allocation.
  uint64_t cache_identity;
  iree_host_size_t entry_point_count;
  iree_hal_amdxdna_kernel_params_t* entry_points;
  // Standalone PDI pool used by in-transaction load_pdi relocations.
  iree_hal_amdxdna_u8_list_t* pdis;
  // Device buffers parallel to `pdis`; allocated on first load_pdi dispatch so
  // unused entry points consume no device memory, then retained so relocated
  // addresses and residency remain stable across dispatches.
  iree_hal_amdxdna_native_buffer_t** pdi_buffers;
  iree_host_size_t pdi_count;
  // Protects lazy PDI-buffer materialization and the cached control-packet
  // context below. Multiple command buffers may be recorded against one
  // executable concurrently.
  iree_slim_mutex_t context_mutex;
  // Shared control-packet context and CU index resolved by the PDI-carrying
  // entry point. Empty-PDI control-packet entry points reuse both.
  iree_hal_amdxdna_native_context_ref_t* context;
  iree_hal_amdxdna_native_c_cu_index_t context_cu_index;
  bool context_cu_index_valid;
} iree_hal_amdxdna_executable;

iree_status_t iree_hal_amdxdna_executable_ensure_pdi_buffer(
    iree_hal_amdxdna_executable* executable, uint32_t pdi_ordinal);

#endif  // IREE_HAL_DRIVERS_AMDXDNA_EXECUTABLE_INTERNAL_H_
