// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_ROCJITSU_USER_QUEUE_H_
#define AMDF_SRC_GPU_UMD_ROCJITSU_USER_QUEUE_H_

#include "libamdf/src/gpu/umd/user_queue.h"

amdf_status_t amdf_rocjitsu_user_queue_create(
    amdf_gpu_umd_device_t* device,
    const amdf_gpu_umd_user_queue_create_info_t* create_info,
    amdf_gpu_umd_user_queue_t** out_queue,
    amdf_gpu_umd_user_queue_result_t* out_result);
amdf_status_t amdf_rocjitsu_user_queue_map(
    amdf_gpu_umd_user_queue_t* queue, amdf_gpu_umd_device_t* producer_device,
    amdf_gpu_umd_user_queue_mapping_t** out_mapping,
    amdf_gpu_umd_user_queue_mapping_result_t* out_result);
amdf_status_t amdf_rocjitsu_user_queue_mapping_destroy(
    amdf_gpu_umd_user_queue_mapping_t* mapping);
amdf_status_t amdf_rocjitsu_user_queue_mapping_flush(
    amdf_gpu_umd_user_queue_mapping_t* mapping);
amdf_status_t amdf_rocjitsu_user_queue_query_status(
    amdf_gpu_umd_user_queue_t* queue, amdf_user_queue_status_t* out_status);
amdf_status_t amdf_rocjitsu_user_queue_query_scratch(
    amdf_gpu_umd_user_queue_t* queue,
    amdf_gpu_aql_scratch_request_t* out_request);
amdf_status_t amdf_rocjitsu_user_queue_resume_scratch(
    amdf_gpu_umd_user_queue_t* queue,
    const amdf_gpu_umd_queue_scratch_t* scratch);
amdf_status_t amdf_rocjitsu_user_queue_trim_scratch(
    amdf_gpu_umd_user_queue_t* queue);
amdf_status_t amdf_rocjitsu_user_queue_set_dispatch_timestamps(
    amdf_gpu_umd_user_queue_t* queue, bool enabled);
amdf_status_t amdf_rocjitsu_user_queue_wait_consumed(
    amdf_gpu_umd_user_queue_t* queue, uint64_t published_index,
    const amdf_wait_deadline_t* deadline);
amdf_status_t amdf_rocjitsu_user_queue_destroy(
    amdf_gpu_umd_user_queue_t* queue);
amdf_status_t amdf_rocjitsu_user_queue_inactivate(
    amdf_gpu_umd_user_queue_t* queue);

#endif  // AMDF_SRC_GPU_UMD_ROCJITSU_USER_QUEUE_H_
