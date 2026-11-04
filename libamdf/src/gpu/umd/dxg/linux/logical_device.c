// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/linux/logical_device.h"

static void amdf_gpu_linux_dxg_logical_device_consume(
    amdf_gpu_linux_dxg_logical_device_t* device) {
  if (device->bridge_adapter.native != NULL) {
    const amdf_status_t status =
        amdf_gpu_dxg_adapter_deinitialize(&device->bridge_adapter);
    // The ABI 13 bridge adapter owns only provider-local host objects. Its
    // close operation has no native failure path and must consume that state.
    if (!amdf_status_is_ok(status)) {
      amdf_abort();
    }
  }
  if (device->native != 0) {
    D3DKMT_DESTROYDEVICE destroy = {
        .hDevice = device->native,
    };
    // Linux DXG removes a valid handle and destroys guest state before its
    // best-effort host notification. Failure therefore indicates an invalid
    // provider-owned handle or transport lifetime, not a retryable teardown.
    if (amdf_linux_dxg_kmt_destroy_device(device->kmt, &destroy) != 0) {
      amdf_abort();
    }
  }
  *device = (amdf_gpu_linux_dxg_logical_device_t){0};
}

amdf_status_t amdf_gpu_linux_dxg_logical_device_initialize(
    const amdf_dxg_bridge_api_t* bridge_api, amdf_platform_endpoint_t* endpoint,
    amdf_allocator_t host_allocator,
    amdf_gpu_linux_dxg_logical_device_t* out_device,
    amdf_dxg_bridge_gpu_properties_t* out_properties) {
  amdf_gpu_linux_dxg_logical_device_t device = {
      .kmt = &endpoint->instance->kmt,
  };
  D3DKMT_CREATEDEVICE create = {
      .hAdapter = endpoint->adapter,
  };
  const NTSTATUS create_status =
      amdf_linux_dxg_kmt_create_device(device.kmt, &create);
  amdf_status_t status = amdf_linux_dxg_kmt_make_status(create_status);
  if (amdf_status_is_ok(status)) {
    device.native = create.hDevice;
    if (device.native == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }

  amdf_dxg_bridge_gpu_properties_t properties = {0};
  if (amdf_status_is_ok(status)) {
    const amdf_dxg_bridge_gpu_adapter_open_info_t open_info = {
        .structure_size = sizeof(open_info),
        .adapter_handle = endpoint->adapter,
        .device_handle = device.native,
        .physical_adapter_index = endpoint->physical_adapter_index,
        .adapter_luid =
            endpoint->info.native_identity.value.windows_adapter.luid,
        .native_transport =
            {
                .context = (void*)device.kmt,
                .invoke = amdf_linux_dxg_kmt_invoke,
            },
    };
    status =
        amdf_gpu_dxg_adapter_initialize(bridge_api, &open_info, host_allocator,
                                        &device.bridge_adapter, &properties);
  }
  if (amdf_status_is_ok(status)) {
    *out_properties = properties;
    *out_device = device;
  } else {
    amdf_gpu_linux_dxg_logical_device_consume(&device);
  }
  return status;
}

void amdf_gpu_linux_dxg_logical_device_deinitialize(
    amdf_gpu_linux_dxg_logical_device_t* device) {
  amdf_gpu_linux_dxg_logical_device_consume(device);
}
