// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_USER_QUEUE_NATIVE_H_
#define AMDF_SRC_GPU_UMD_KFD_USER_QUEUE_NATIVE_H_

#include <drm/amdgpu_drm.h>
#include <linux/kfd_ioctl.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "amdf/amdf.h"
#include "libamdf/src/gpu/umd/kfd/buffer.h"
#include "libamdf/src/gpu/umd/kfd/reset_monitor.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Complete semantic result of one KFD queue-destroy attempt.
typedef struct amdf_gpu_kfd_user_queue_destroy_result_t {
  // Native status returned by queue destruction.
  amdf_status_t status;
  // Whether KFD consumed the supplied queue identifier despite `status`.
  bool identifier_consumed;
} amdf_gpu_kfd_user_queue_destroy_result_t;

// Translates one native destroy status into the KFD identifier-ownership
// result established by the queue-manager UAPI.
amdf_gpu_kfd_user_queue_destroy_result_t
amdf_gpu_kfd_user_queue_classify_destroy_status(amdf_status_t status);

// Native operations beneath the transactional KFD queue owner.
typedef struct amdf_gpu_kfd_user_queue_native_api_t {
  // Opaque state passed to every operation.
  void* user_data;
  // Creates one mapped KFD allocation. Failure leaves both outputs unchanged.
  amdf_status_t (*buffer_create)(
      void* user_data, amdf_gpu_umd_device_t* device,
      const amdf_gpu_kfd_buffer_create_info_t* create_info,
      amdf_gpu_kfd_buffer_t** out_buffer,
      amdf_gpu_kfd_buffer_result_t* out_result);
  // Releases one mapped KFD allocation, retaining it on failure.
  amdf_status_t (*buffer_destroy)(void* user_data,
                                  amdf_gpu_kfd_buffer_t* buffer);
  // Frees only host bookkeeping, leaving native backing and mappings intact
  // after terminal construction rollback failure.
  void (*buffer_abandon)(void* user_data, amdf_gpu_kfd_buffer_t* buffer);
  // Creates one KFD queue and publishes native output fields only on success.
  amdf_status_t (*queue_create)(
      void* user_data, amdf_gpu_umd_device_t* device,
      struct kfd_ioctl_create_queue_args* inout_arguments);
  // Attempts to destroy one KFD queue and classifies identifier ownership.
  amdf_gpu_kfd_user_queue_destroy_result_t (*queue_destroy)(
      void* user_data, amdf_gpu_umd_device_t* device,
      uint32_t queue_identifier);
  // Maps one page-aligned KFD doorbell aperture. Failure leaves the output
  // unchanged.
  amdf_status_t (*doorbell_map)(void* user_data, amdf_gpu_umd_device_t* device,
                                uint64_t native_byte_offset, size_t byte_length,
                                void** out_mapping);
  // Releases one complete KFD doorbell aperture, retaining it on failure.
  amdf_status_t (*doorbell_unmap)(void* user_data, void* mapping,
                                  size_t byte_length);
  // Samples the fault cache of the device's own render-file VM, not global
  // hardware state. Failure leaves the output unchanged.
  amdf_status_t (*vm_fault_query)(
      void* user_data, amdf_gpu_umd_device_t* device,
      struct drm_amdgpu_info_gpuvm_fault* out_fault);
  // Samples the device-epoch reset observer. Failure leaves the output
  // unchanged.
  amdf_status_t (*reset_query)(void* user_data, amdf_gpu_umd_device_t* device,
                               amdf_gpu_kfd_reset_state_t* out_state);
} amdf_gpu_kfd_user_queue_native_api_t;

// Returns the production KFD queue operation table.
const amdf_gpu_kfd_user_queue_native_api_t*
amdf_gpu_kfd_user_queue_default_native_api(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_KFD_USER_QUEUE_NATIVE_H_
