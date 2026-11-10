// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBHRX_SRC_LIBHRX_QUEUE_H_
#define LIBHRX_SRC_LIBHRX_QUEUE_H_

#include "hrx_runtime.h"
#include "iree/base/api.h"
#include "iree/hal/queue.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hrx_queue_s {
  // Public references; provisioned handles retain a device-owned base
  // reference.
  iree_atomic_ref_count_t ref_count;
  // Borrowed parent, which must outlive the queue.
  hrx_device_t device;
  // Exact native queue; owned for acquisitions, borrowed for provisioned
  // queues.
  iree_hal_queue_t* hal_queue;
} hrx_queue_s;

// Initializes the immutable provisioned-queue handles in one device-owned
// allocation. The native device and queue inventory are already constructed.
iree_status_t hrx_device_initialize_queues(hrx_device_t device);

// Releases the device-owned table after callers have released their references.
// Provisioned handles cannot reach zero references before this point; only
// separately acquired handles use hrx_queue_release's final destruction path.
void hrx_device_deinitialize_queues(hrx_device_t device);

// Validates operation roles at an HRX submission boundary. The caller has
// already established that the supplied queue is non-NULL.
iree_status_t hrx_queue_validate_roles(hrx_queue_t queue,
                                       hrx_queue_role_flags_t required_roles);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_LIBHRX_QUEUE_H_
