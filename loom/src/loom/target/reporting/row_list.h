// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_REPORTING_ROW_LIST_H_
#define LOOM_TARGET_REPORTING_ROW_LIST_H_

#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Releases every storage block owned by |list| and resets it to empty.
void loom_target_compile_report_row_list_deinitialize(
    iree_allocator_t allocator, loom_target_compile_report_row_list_t* list);

// Appends one |row_size|-byte row to |list|.
iree_status_t loom_target_compile_report_row_list_append(
    loom_target_compile_report_row_list_t* list, iree_host_size_t row_size,
    iree_allocator_t allocator, const void* row);

// Appends every row in |source| to |target|.
iree_status_t loom_target_compile_report_row_list_append_all(
    loom_target_compile_report_row_list_t* target,
    const loom_target_compile_report_row_list_t* source,
    iree_host_size_t row_size, iree_allocator_t allocator);

// Initializes |target| as an owned copy of |source|.
iree_status_t loom_target_compile_report_row_list_clone(
    const loom_target_compile_report_row_list_t* source,
    iree_host_size_t row_size, iree_allocator_t allocator,
    loom_target_compile_report_row_list_t* target);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TARGET_REPORTING_ROW_LIST_H_
