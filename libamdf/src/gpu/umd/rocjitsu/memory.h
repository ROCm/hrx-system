// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_ROCJITSU_MEMORY_H_
#define AMDF_SRC_GPU_UMD_ROCJITSU_MEMORY_H_

#include "libamdf/src/gpu/umd/memory.h"

amdf_status_t amdf_rocjitsu_device_query_memory_profile(
    amdf_gpu_umd_device_t* device, uint32_t ordinal,
    amdf_memory_native_profile_t* out_profile);
amdf_status_t amdf_rocjitsu_memory_prepare(
    amdf_gpu_umd_device_t* device, uint32_t peer_count,
    amdf_gpu_umd_device_t* const* peers,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_gpu_umd_memory_t** state, amdf_gpu_umd_memory_result_t* out_result);
amdf_status_t amdf_rocjitsu_memory_destroy(amdf_gpu_umd_memory_t* memory);
uint64_t amdf_rocjitsu_memory_remaining_backing_byte_length(
    const amdf_gpu_umd_memory_t* memory);
void amdf_rocjitsu_memory_abandon(amdf_gpu_umd_memory_t* memory);
amdf_status_t amdf_rocjitsu_memory_map(
    amdf_gpu_umd_memory_t* memory,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info,
    amdf_gpu_umd_host_mapping_t** out_mapping,
    amdf_gpu_umd_host_mapping_result_t* out_result);
amdf_status_t amdf_rocjitsu_host_mapping_cache_control(
    amdf_gpu_umd_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
    uint64_t byte_offset, uint64_t byte_length);
void amdf_rocjitsu_host_mapping_destroy(amdf_gpu_umd_host_mapping_t* mapping);

#endif  // AMDF_SRC_GPU_UMD_ROCJITSU_MEMORY_H_
