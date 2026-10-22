// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_D3D12_QUEUE_TEST_NATIVE_H_
#define AMDF_SRC_MEDIA_D3D12_QUEUE_TEST_NATIVE_H_

#include <d3d12.h>
#include <d3d12video.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A COM dependency, implemented with the SDK's C vtables so unused methods
// need no forwarding boilerplate. No native device or work is created.
typedef struct amdf_media_queue_test_native_t {
  // Factory borrowed by the media device under test.
  ID3D12Device3* device;
  // Complete recording supplied to the queue under test.
  ID3D12VideoDecodeCommandList* command;
  // Cold progress construction failure, after native queue construction.
  HRESULT fence_create_status;
  // Result of a native signal attempt after execution or accepted waits.
  HRESULT signal_status;
  // Independently observed device-removal status.
  HRESULT removed_status;
  // One-based native wait call to reject, or zero to accept every wait.
  uint32_t reject_wait;
  // Number of native waits attempted.
  uint32_t wait_count;
  // Number of native command executions accepted.
  uint32_t execute_count;
  // Number of native progress signals attempted.
  uint32_t signal_count;
  // Value supplied to the last progress signal attempt.
  uint64_t signaled;
  // Completion observed independently of publication's return status.
  uint64_t completed;
  // Number of native completion queries.
  uint32_t completion_query_count;
  // Number of independent native device-state queries.
  uint32_t removal_query_count;
  // Live native queue references.
  uint32_t queue_references;
  // Live native fence references, including cold dependency imports.
  uint32_t fence_references;
  // Live native device references.
  uint32_t device_references;
  // Live native command-list references.
  uint32_t command_references;
} amdf_media_queue_test_native_t;

amdf_media_queue_test_native_t* amdf_media_queue_test_native_create(void);
void amdf_media_queue_test_native_destroy(
    amdf_media_queue_test_native_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_D3D12_QUEUE_TEST_NATIVE_H_
