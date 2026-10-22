// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_D3D12_QUEUE_H_
#define AMDF_SRC_MEDIA_D3D12_QUEUE_H_

#include "libamdf/src/media/d3d12/device.h"
#include "libamdf/src/platform/windows/monitored_fence.h"

namespace amdf::media::d3d12 {

struct Queue {
  // Core lifetime, capabilities, and checked-progress interface.
  amdf_kernel_queue_t base;
  // Native ordered decode execution queue.
  ComPtr<ID3D12CommandQueue> native;
  // Native progress used for every publication and notification.
  ComPtr<ID3D12Fence> fence;
  // KMT-origin progress when external synchronization was explicitly requested.
  amdf_windows_monitored_fence_t shared_progress;
  // Exclusive host publication claim, independent of observation and waiting.
  amdf_atomic_uint32_t publishing;
  // Greatest accepted native prefix, including failed partial publication.
  amdf_atomic_uint64_t submitted;
  // Greatest independently checked completed prefix.
  amdf_atomic_uint64_t retired;
  // First publication failure after native acceptance, separate from
  // retirement.
  amdf_atomic_uint64_t terminal_status;
  // Cold reusable host wait event; publication never touches it.
  HANDLE wait_event;
  // Serializes event registration/waits while respecting each caller's
  // deadline.
  SRWLOCK wait_lock;
  // Earliest outstanding notification to the reusable wait event, or zero.
  uint64_t wait_event_submission;
};

amdf_status_t AMDF_CALL
QueueCreate(amdf_device_t* device,
            const amdf_media_kernel_queue_create_info_t* create_info,
            amdf_kernel_queue_t** out_queue);
amdf_status_t AMDF_CALL
QueueSubmit(amdf_kernel_queue_t* queue,
            const amdf_media_kernel_queue_submission_info_t* submission_info,
            uint64_t* accepted_submission);

// Samples native completion and device-removal state without a host wait.
// Native failure is cached separately and never advances checked retirement.
void RefreshRetirement(Queue* queue);

// Reads retained queue/device failure without a native query.
amdf_status_t QueueTerminalStatus(const Queue* queue);

}  // namespace amdf::media::d3d12

#endif  // AMDF_SRC_MEDIA_D3D12_QUEUE_H_
