// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_COMMON_KERNEL_ARGUMENTS_H_
#define HRX_BINDING_COMMON_KERNEL_ARGUMENTS_H_

#include "common/internal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Validates a caller-supplied native kernarg byte image before it is copied or
// dispatched. The reflected direct-argument extent covers meaningful native
// parameter bytes; callers may omit ABI tail padding but may not truncate that
// extent. The declared span length remains authoritative after validation.
iree_status_t iree_hal_streaming_validate_prepacked_kernel_arguments(
    const iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params);

// Unpacks a packed kernel parameter buffer into a constant buffer and binding
// list. Some dispatches may use raw device buffer pointers and others may use
// bindings that can be resolved to HAL buffers.
// Callers must ensure sufficient storage in |out_constants| and |out_bindings|
// based on the symbol constant size and binding count.
// Synchronization: none (data packing utility).
iree_status_t iree_hal_streaming_unpack_parameters(
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_parameter_info_t* parameters,
    const void* parameter_buffer, void* out_constants,
    iree_hal_buffer_ref_list_t* out_bindings);

// Packs a pointer-array launch parameter list into a fully initialized native
// ABI byte image. Each element in |parameter_list| points at the bytes of the
// corresponding argument value; device pointer bytes need not have host pointer
// alignment or an effective host pointer type. Copy and resolve operation
// partitions are merged by their globally unique source ordinals.
// |out_constants_size| receives the validated image length described by
// |parameters|.
// Synchronization: none (data packing utility).
iree_status_t iree_hal_streaming_pack_raw_argument_list(
    const iree_hal_streaming_parameter_info_t* parameters,
    void** parameter_list, void* out_constants,
    iree_host_size_t* out_constants_size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_COMMON_KERNEL_ARGUMENTS_H_
