// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CHANNEL_TRANSFER_H_
#define IREE_HAL_CHANNEL_TRANSFER_H_

#include "iree/hal/buffer.h"
#include "iree/hal/channel.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/pool.h"
#include "iree/hal/queue.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Access to the local buffer, independent of the initiator of each exchange.
typedef enum iree_hal_channel_transfer_mode_e {
  IREE_HAL_CHANNEL_TRANSFER_MODE_SEND = 0,
  IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE = 1,
} iree_hal_channel_transfer_mode_t;

// Immutable description of a reusable directed exchange. The matching peer
// prepares the opposite mode with the same tag. Tags are scoped by channel,
// source rank and target rank; they are independent of local queue identity.
typedef struct iree_hal_channel_transfer_params_t {
  // Whether the local buffer supplies or receives bytes.
  iree_hal_channel_transfer_mode_t mode;
  // Exact other participant in this channel's rank namespace.
  int32_t peer_rank;
  // Application-selected identity, unique within this directed peer edge.
  uint32_t tag;
  // Shared storage policy retained by this preparation. Execution ranges must
  // inherit this pool's exact memory contract; no payload range is reserved.
  iree_hal_pool_t* pool;
  // Maximum number of bytes admitted in one exchange.
  iree_device_size_t capacity;
} iree_hal_channel_transfer_params_t;

// Prepared communication resources, independent of queue or device initiation.
// Preparation qualifies local access and reserves bounded progress storage.
// It performs no payload allocation or transfer. Native registration belongs
// to the backing owner or an explicit target-specific publication. Such a
// device publication must retain this owner through both device access and all
// network operations issued through its descriptors. Dispatch completion alone
// does not establish network retirement.
typedef struct iree_hal_channel_transfer_t iree_hal_channel_transfer_t;

// Prepares |params| once for repeated queue send/receive operations. The
// transfer retains the shared pool without reserving tensor capacity. Each
// admitted exchange retains its own range through actual transfer retirement.
// Unsupported memory or progress routes fail instead of staging implicitly.
// The caller orders all foreign uses before queue_dealloca: retaining a buffer
// object does not extend a queue allocation epoch.
IREE_API_EXPORT iree_status_t iree_hal_channel_prepare_transfer(
    iree_hal_channel_t* channel,
    const iree_hal_channel_transfer_params_t* params,
    iree_hal_channel_transfer_t** out_transfer);

IREE_API_EXPORT void iree_hal_channel_transfer_retain(
    iree_hal_channel_transfer_t* transfer);
IREE_API_EXPORT void iree_hal_channel_transfer_release(
    iree_hal_channel_transfer_t* transfer);

//===----------------------------------------------------------------------===//
// iree_hal_channel_transfer_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_channel_transfer_vtable_t {
  // Releases the prepared resource after its last caller/operation/publication.
  void(IREE_API_PTR* destroy)(iree_hal_channel_transfer_t* transfer);
  // Admits a validated exchange, retaining all captured dependencies until its
  // actual callbacks retire. Rejection captures nothing and signals nothing.
  iree_status_t(IREE_API_PTR* submit)(
      iree_hal_channel_transfer_t* transfer, iree_hal_queue_t* queue,
      iree_hal_semaphore_list_t wait_semaphore_list,
      iree_hal_semaphore_list_t signal_semaphore_list, uint64_t sequence,
      iree_hal_buffer_ref_t buffer);
} iree_hal_channel_transfer_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_channel_transfer_vtable_t);

struct iree_hal_channel_transfer_t {
  // Base HAL resource; must be at offset zero.
  iree_hal_resource_t resource;
  // Retained channel owning the route, qualified backing and native resources.
  iree_hal_channel_t* channel;
  // Device retained by the channel, used to qualify exact local queues.
  iree_hal_device_t* device;
  // Local families admitted by the prepared route.
  iree_hal_queue_family_affinity_t queue_family_affinity;
  // Immutable, validated access and matching information.
  iree_hal_channel_transfer_params_t params;
};

IREE_API_EXPORT void iree_hal_channel_transfer_initialize(
    const iree_hal_channel_transfer_vtable_t* vtable,
    iree_hal_channel_t* channel, iree_hal_device_t* device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    const iree_hal_channel_transfer_params_t* params,
    iree_hal_channel_transfer_t* out_transfer);
IREE_API_EXPORT void iree_hal_channel_transfer_deinitialize(
    iree_hal_channel_transfer_t* transfer);
IREE_API_EXPORT void iree_hal_channel_transfer_destroy(
    iree_hal_channel_transfer_t* transfer);

// Validates an externally supplied execution range against its prepared
// capacity and exact pool contract. Does not inspect native binding values:
// queue dependencies may publish them after submission.
IREE_API_EXPORT iree_status_t iree_hal_channel_transfer_validate_buffer(
    const iree_hal_channel_transfer_t* transfer, iree_hal_buffer_ref_t buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_CHANNEL_TRANSFER_H_
