// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_PARAMETER_INDEX_PROVIDER_H_
#define IREE_IO_PARAMETER_INDEX_PROVIDER_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Reasonable default for the `max_concurrent_operations` parameter.
#define IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS 16

// Creates a parameter provider serving from the provided |index|.
// As parameters are operated on their files will be registered with the devices
// they are used on and cached for future requests.
//
// |max_concurrent_operations| can be used to limit how many file operations as
// part of a gather or scatter are allowed to be in-flight at a time. A lower
// number can reduce system resource requirements during the operation (less
// transient memory required, etc) while increasing latency (lower I/O
// utilization). IREE_HOST_SIZE_MAX permits every span in a group to be
// in flight. The limit applies independently to each synchronized group in
// gather_batch/scatter_batch; one group's waits never gate another group.
IREE_API_EXPORT iree_status_t iree_io_parameter_index_provider_create(
    iree_string_view_t scope, iree_io_parameter_index_t* index,
    iree_host_size_t max_concurrent_operations, iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider);

// Resolves one indexed parameter to its complete imported HAL file for reading
// and absolute byte range within that file. |provider| must have been created
// by iree_io_parameter_index_provider_create. This cold lookup reuses the
// provider's existing file cache and reads or allocates no payload storage.
// Importing a previously unused file may allocate native handle metadata.
// Synthetic entries without backing files return UNAVAILABLE.
//
// On success, the caller owns |out_file| independently of the provider/index;
// its device and progress owner must outlive use, as for iree_hal_file_import.
// Subsequent byte operations use that retained handle and do not resolve the
// parameter again. Native transfer requirements come from the returned file;
// the parameter origin itself need not satisfy those requirements. On failure,
// all outputs are zero/null. Access is qualified for the queue's exact family.
IREE_API_EXPORT iree_status_t iree_io_parameter_index_provider_resolve_file(
    iree_io_parameter_provider_t* provider, iree_hal_queue_t* queue,
    iree_string_view_t key, iree_hal_file_t** out_file,
    uint64_t* out_file_offset, uint64_t* out_length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_PARAMETER_INDEX_PROVIDER_H_
