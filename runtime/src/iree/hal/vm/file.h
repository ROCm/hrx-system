// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_VM_FILE_H_
#define IREE_HAL_VM_FILE_H_

#include "iree/hal/queue.h"
#include "iree/hal/vm/types.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates the optional "hal.file" module on an exact caller-selected queue.
// Both read_indirect and write_indirect have the following machine signature:
//   (span<ptr<hal.file>, read> files, span<i64, read> file_windows,
//    span<ptr<hal.buffer>, read> buffers, span<i64, read> buffer_windows,
//    ptr<hal.fence> wait, ptr<hal.buffer> records, i64 records_offset,
//    i64 records_length, ptr<hal.buffer> count, i64 count_offset) -> hal.fence
//
// Each window span contains one pair of unsigned byte offset/length values per
// object. File lengths are explicit; buffer lengths may be UINT64_MAX to select
// the remaining buffer. Both directions use the native HAL's finite 64-byte
// record encoding, including binding indices and strided geometry. The count
// is a little-endian u64 consumed only after the optional wait fence succeeds.
// Zero counts still preserve the dependency and visibility contract.
//
// Arrays are borrowed only during the synchronous call. Accepted native work
// captures every file and buffer independently of this module and returns an
// independently owned completion fence. A successful signal permits consumers
// and descriptor reuse; dropping or independently failing a fence does not
// cancel work or retire its resources. The hosting device group, pools, and
// external backing retain their ordinary native lifetime obligations.
//
// Retains the queue and its device. Canonical reference descriptors and their
// destructor code outlive the module and any escaped results. No execution
// frontier, cache policy, file opening, or host wait is implicit in these
// calls.
IREE_API_EXPORT iree_status_t iree_hal_vm_file_module_create(
    const iree_hal_vm_types_t* types, iree_hal_queue_t* queue,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_VM_FILE_H_
