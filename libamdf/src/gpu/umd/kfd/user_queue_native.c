// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/gpu/umd/kfd/user_queue_native.h"

#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "libamdf/src/platform/linux/file.h"

amdf_gpu_kfd_user_queue_destroy_result_t
amdf_gpu_kfd_user_queue_classify_destroy_status(amdf_status_t status) {
  const bool failed_after_consumption =
      amdf_status_domain(status) == AMDF_STATUS_DOMAIN_ERRNO &&
      (amdf_status_code(status) == ETIME || amdf_status_code(status) == EIO);
  return (amdf_gpu_kfd_user_queue_destroy_result_t){
      .status = status,
      .identifier_consumed =
          amdf_status_is_ok(status) || failed_after_consumption,
  };
}

static amdf_status_t amdf_gpu_kfd_user_queue_buffer_create(
    void* user_data, amdf_gpu_umd_device_t* device,
    const amdf_gpu_kfd_buffer_create_info_t* create_info,
    amdf_gpu_kfd_buffer_t** out_buffer,
    amdf_gpu_kfd_buffer_result_t* out_result) {
  (void)user_data;
  return amdf_gpu_kfd_buffer_create(device, create_info, out_buffer,
                                    out_result);
}

static amdf_status_t amdf_gpu_kfd_user_queue_buffer_destroy(
    void* user_data, amdf_gpu_kfd_buffer_t* buffer) {
  (void)user_data;
  return amdf_gpu_kfd_buffer_destroy(buffer);
}

static void amdf_gpu_kfd_user_queue_buffer_abandon(
    void* user_data, amdf_gpu_kfd_buffer_t* buffer) {
  (void)user_data;
  amdf_gpu_kfd_buffer_abandon(buffer);
}

static amdf_status_t amdf_gpu_kfd_user_queue_create_native(
    void* user_data, amdf_gpu_umd_device_t* device,
    struct kfd_ioctl_create_queue_args* inout_arguments) {
  (void)user_data;
  struct kfd_ioctl_create_queue_args arguments = *inout_arguments;
  if (ioctl(device->descriptor, AMDKFD_IOC_CREATE_QUEUE, &arguments) != 0) {
    return amdf_linux_error(errno);
  }
  *inout_arguments = arguments;
  return AMDF_STATUS_OK;
}

static amdf_gpu_kfd_user_queue_destroy_result_t
amdf_gpu_kfd_user_queue_destroy_native(void* user_data,
                                       amdf_gpu_umd_device_t* device,
                                       uint32_t queue_identifier) {
  (void)user_data;
  struct kfd_ioctl_destroy_queue_args arguments = {
      .queue_id = queue_identifier,
  };
  if (ioctl(device->descriptor, AMDKFD_IOC_DESTROY_QUEUE, &arguments) == 0) {
    return amdf_gpu_kfd_user_queue_classify_destroy_status(AMDF_STATUS_OK);
  }
  return amdf_gpu_kfd_user_queue_classify_destroy_status(
      amdf_linux_error(errno));
}

static amdf_status_t amdf_gpu_kfd_user_queue_doorbell_map(
    void* user_data, amdf_gpu_umd_device_t* device, uint64_t native_byte_offset,
    size_t byte_length, void** out_mapping) {
  (void)user_data;
  void* mapping = mmap(NULL, byte_length, PROT_READ | PROT_WRITE, MAP_SHARED,
                       device->descriptor, (off_t)native_byte_offset);
  if (mapping == MAP_FAILED) return amdf_linux_error(errno);
  *out_mapping = mapping;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_kfd_user_queue_doorbell_unmap(
    void* user_data, void* mapping, size_t byte_length) {
  (void)user_data;
  return munmap(mapping, byte_length) == 0 ? AMDF_STATUS_OK
                                           : amdf_linux_error(errno);
}

static amdf_status_t amdf_gpu_kfd_user_queue_reset_query(
    void* user_data, amdf_gpu_umd_device_t* device,
    amdf_gpu_kfd_reset_state_t* out_state) {
  (void)user_data;
  return amdf_gpu_kfd_reset_monitor_query(&device->reset_monitor, out_state);
}

static amdf_status_t amdf_gpu_kfd_user_queue_vm_fault_query(
    void* user_data, amdf_gpu_umd_device_t* device,
    struct drm_amdgpu_info_gpuvm_fault* out_fault) {
  (void)user_data;
  struct drm_amdgpu_info_gpuvm_fault fault = {0};
  struct drm_amdgpu_info query = {
      .return_pointer = (uintptr_t)&fault,
      .return_size = sizeof(fault),
      .query = AMDGPU_INFO_GPUVM_FAULT,
  };
  if (ioctl(device->render_descriptor, DRM_IOCTL_AMDGPU_INFO, &query) != 0) {
    return amdf_linux_error(errno);
  }
  *out_fault = fault;
  return AMDF_STATUS_OK;
}

static const amdf_gpu_kfd_user_queue_native_api_t
    amdf_gpu_kfd_user_queue_native_api = {
        .buffer_create = amdf_gpu_kfd_user_queue_buffer_create,
        .buffer_destroy = amdf_gpu_kfd_user_queue_buffer_destroy,
        .buffer_abandon = amdf_gpu_kfd_user_queue_buffer_abandon,
        .queue_create = amdf_gpu_kfd_user_queue_create_native,
        .queue_destroy = amdf_gpu_kfd_user_queue_destroy_native,
        .doorbell_map = amdf_gpu_kfd_user_queue_doorbell_map,
        .doorbell_unmap = amdf_gpu_kfd_user_queue_doorbell_unmap,
        .vm_fault_query = amdf_gpu_kfd_user_queue_vm_fault_query,
        .reset_query = amdf_gpu_kfd_user_queue_reset_query,
};

const amdf_gpu_kfd_user_queue_native_api_t*
amdf_gpu_kfd_user_queue_default_native_api(void) {
  return &amdf_gpu_kfd_user_queue_native_api;
}
