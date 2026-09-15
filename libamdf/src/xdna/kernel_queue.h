// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_KERNEL_QUEUE_H_
#define AMDF_SRC_XDNA_KERNEL_QUEUE_H_

#include "amdf/xdna.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Acquires one kernel-mediated XDNA queue from a schedulable context.
amdf_status_t AMDF_CALL amdf_xdna_kernel_queue_create(
    amdf_xdna_context_t* context,
    const amdf_xdna_kernel_queue_create_info_t* create_info,
    amdf_kernel_queue_t** out_queue);

// Publishes one bounded set of immutable XDNA commands.
amdf_status_t AMDF_CALL amdf_xdna_kernel_queue_submit(
    amdf_kernel_queue_t* queue,
    const amdf_xdna_kernel_queue_submission_info_t* submission_info,
    uint64_t* out_submission);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_KERNEL_QUEUE_H_
