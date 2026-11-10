// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBHRX_SRC_LIBHRX_DEVICE_GROUP_H_
#define LIBHRX_SRC_LIBHRX_DEVICE_GROUP_H_

#include "device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hrx_device_group_s {
  // External group references, including public retained device views.
  iree_atomic_ref_count_t ref_count;
  // Retained shared execution services; released after all native devices.
  hrx_runtime_t runtime;
  // Retained native devices and their immutable topology.
  iree_hal_device_group_t* hal_group;
  // Number of initialized inline device views.
  size_t device_count;
  // Inline views, allocated with the group and never independently freed.
  hrx_device_s devices[];
} hrx_device_group_s;

// Private frontend entry for native creation extensions. The frontend keeps
// provider state alive through final group destruction. The public constructor
// passes NULL; URI options carry public driver/runtime feature requests.
iree_status_t hrx_device_group_create_with_extensions(
    hrx_runtime_t runtime, size_t device_count,
    const hrx_device_create_params_t* devices,
    const iree_hal_device_create_params_extension_t* extensions,
    iree_hal_device_runtime_feature_flags_t runtime_features,
    hrx_device_group_t* out_group);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_LIBHRX_DEVICE_GROUP_H_
