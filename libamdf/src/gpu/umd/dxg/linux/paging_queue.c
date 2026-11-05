// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/linux/paging_queue.h"

amdf_status_t amdf_gpu_linux_dxg_paging_queue_consume(
    amdf_gpu_linux_dxg_paging_queue_t* queue) {
  amdf_status_t status = AMDF_STATUS_OK;
  if (queue->native != 0) {
    D3DDDI_DESTROYPAGINGQUEUE destroy = {
        .hPagingQueue = queue->native,
    };
    status = amdf_linux_dxg_kmt_make_status(
        amdf_linux_dxg_kmt_destroy_paging_queue(queue->kmt, &destroy));
  }
  *queue = (amdf_gpu_linux_dxg_paging_queue_t){0};
  return status;
}

amdf_status_t amdf_gpu_linux_dxg_paging_queue_initialize(
    const amdf_gpu_linux_dxg_logical_device_t* logical_device,
    uint32_t physical_adapter_index,
    amdf_gpu_linux_dxg_paging_queue_t* out_queue) {
  D3DKMT_CREATEPAGINGQUEUE create = {
      .hDevice = logical_device->native,
      .Priority = D3DDDI_PAGINGQUEUE_PRIORITY_NORMAL,
      .PhysicalAdapterIndex = physical_adapter_index,
  };
  amdf_status_t status = amdf_linux_dxg_kmt_make_status(
      amdf_linux_dxg_kmt_create_paging_queue(logical_device->kmt, &create));
  amdf_gpu_linux_dxg_paging_queue_t queue = {
      .kmt = logical_device->kmt,
  };
  if (amdf_status_is_ok(status)) {
    queue.native = create.hPagingQueue;
    queue.synchronization_object = create.hSyncObject;
    queue.progress_fence =
        (const volatile uint64_t*)create.FenceValueCPUVirtualAddress;
    if (queue.native == 0 || queue.synchronization_object == 0 ||
        queue.progress_fence == NULL) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }
  if (amdf_status_is_ok(status)) {
    *out_queue = queue;
  } else {
    const amdf_status_t consume_status =
        amdf_gpu_linux_dxg_paging_queue_consume(&queue);
    if (!amdf_status_is_ok(consume_status)) {
      status = consume_status;
    }
  }
  return status;
}
