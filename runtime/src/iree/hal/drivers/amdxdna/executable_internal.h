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
#include "iree/hal/drivers/amdxdna/direct_command_buffer_planning.h"
#include "iree/hal/drivers/amdxdna/executable.h"
#include "iree/hal/drivers/amdxdna/native.h"

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
  iree_hal_amdxdna_u8_list_t pdi;
  // AXLF/xclbin context wrapper from amdxdna-xclbin-fb for native
  // drivers that consume an xclbin-shaped context blob.
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
  iree_hal_amdxdna_write32_constant_patch_list_t* constant_patch_runlist;
  iree_host_size_t constant_patch_runlist_count;
  iree_string_view_t kernel_name;
  uint32_t n_reconfigure_runs;
  uint32_t n_pdi_loads;
  IREE_TRACE(iree_string_view_t source_filename;)
  IREE_TRACE(uint32_t source_line;)
} iree_hal_amdxdna_kernel_params_t;

typedef struct iree_hal_amdxdna_executable {
  // Abstract resource used for injecting reference counting and vtable; must be
  // at offset 0.
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  // Process-unique identity used by native command caches. Unlike the object
  // address this is never reused after destruction, so cached immutable
  // entry-point descriptors cannot alias a later executable allocation.
  uint64_t cache_identity;
  iree_host_size_t entry_point_count;
  iree_hal_amdxdna_kernel_params_t* entry_points;
  // Protects the cached control-packet context below. Multiple command buffers
  // may be recorded against one executable concurrently.
  iree_slim_mutex_t context_mutex;
  // Shared control-packet context and CU index resolved by the PDI-carrying
  // entry point. Empty-PDI control-packet entry points reuse both.
  iree_hal_amdxdna_native_context_ref_t* context;
  iree_hal_amdxdna_native_c_cu_index_t context_cu_index;
  bool context_cu_index_valid;
} iree_hal_amdxdna_executable;

#endif  // IREE_HAL_DRIVERS_AMDXDNA_EXECUTABLE_INTERNAL_H_
