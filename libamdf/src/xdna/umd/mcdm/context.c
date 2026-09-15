// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/context.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/xdna/umd/mcdm/context.h"
#include "libamdf/src/xdna/umd/mcdm/kernel_execution.h"
#include "libamdf/src/xdna/umd/mcdm/legacy_context.h"

amdf_status_t amdf_xdna_umd_context_destroy(amdf_xdna_umd_context_t* context) {
  if (context->kernel_execution != NULL) {
    const amdf_status_t status =
        amdf_windows_xdna_kernel_execution_prepare_context_destroy(
            context->kernel_execution);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  if (context->handle != 0) {
    D3DKMT_DESTROYCONTEXT destroy = {0};
    destroy.hContext = context->handle;
    const amdf_status_t status =
        amdf_kmt_make_status(context->device->kmt->destroy_context(&destroy));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    context->handle = 0;
  }
  if (context->kernel_execution != NULL) {
    const amdf_status_t status =
        amdf_windows_xdna_kernel_execution_destroy(context->kernel_execution);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    context->kernel_execution = NULL;
  }
  const amdf_allocator_t host_allocator = context->device->host_allocator;
  amdf_free(host_allocator, context);
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_xdna_umd_context_create(
    amdf_xdna_umd_device_t* device,
    const amdf_xdna_context_create_info_t* create_info,
    amdf_xdna_umd_context_t** out_context,
    amdf_xdna_umd_context_result_t* out_result) {
  const amdf_xdna_endpoint_profile_t* profile = device->profile;
  // Device/paging resources also serve ordinary memory and do not establish
  // support for this context's interpreter bootstrap or private wire ABI.
  if ((profile->execution_capabilities &
       AMDF_XDNA_EXECUTION_CAPABILITY_TRANSACTION_INTERPRETER_V1) == 0 ||
      device->kmt->create_context_virtual == NULL ||
      device->kmt->destroy_context == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if ((create_info->acceptable_scheduling_modes &
       AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED) == 0 ||
      create_info->physical_column_origin !=
          AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }

  const amdf_windows_xdna_native_abi_t* native_abi = NULL;
  amdf_status_t status = amdf_windows_xdna_native_abi_query(
      device->kmt, device->adapter, &native_abi);
  if (!amdf_status_is_ok(status)) return status;
  uint8_t* context_data = NULL;
  uint32_t context_data_size = 0;
  status = amdf_windows_xdna_legacy_context_build(
      native_abi, profile->bootstrap, create_info->logical_column_count,
      profile->info->array.column_origin, device->host_allocator, &context_data,
      &context_data_size);
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  amdf_xdna_umd_context_t* context = NULL;
  status = amdf_calloc(device->host_allocator, sizeof(*context),
                       amdf_alignof(amdf_xdna_umd_context_t), (void**)&context);
  if (!amdf_status_is_ok(status)) {
    amdf_free(device->host_allocator, context_data);
    return status;
  }
  context->device = device;
  context->native_abi = native_abi;

  D3DKMT_CREATECONTEXTVIRTUAL create = {0};
  create.hDevice = device->device;
  create.NodeOrdinal = 0;
  create.EngineAffinity = 1;
  create.Flags.HwQueueSupported = 1;
  create.pPrivateDriverData = context_data;
  create.PrivateDriverDataSize = context_data_size;
  create.ClientHint = (D3DKMT_CLIENTHINT)25;
  status = amdf_kmt_make_status(device->kmt->create_context_virtual(&create));
  if (amdf_status_is_ok(status)) {
    context->handle = create.hContext;
    if (context->handle == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    } else {
      context->command_aperture_cookie =
          amdf_windows_xdna_legacy_context_query_command_aperture_cookie(
              native_abi, context_data);
      if (context->command_aperture_cookie > UINT8_MAX) {
        status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
      }
    }
  }
  amdf_free(device->host_allocator, context_data);

  if (amdf_status_is_ok(status)) {
    status = amdf_windows_xdna_kernel_execution_create(
        context, &context->kernel_execution);
  }
  if (amdf_status_is_ok(status)) {
    amdf_xdna_umd_context_result_t result = {0};
    result.id.words[0] = (uintptr_t)context;
    result.id.words[1] = ((uint64_t)context->handle << 32) | device->device;
    result.scheduling_mode = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
    *out_result = result;
    *out_context = context;
  } else {
    const amdf_status_t release_status = amdf_xdna_umd_context_destroy(context);
    if (!amdf_status_is_ok(release_status)) {
      // Failed construction has no execution storage or accepted work. A
      // native handle leak does not require retaining this host bookkeeping.
      amdf_free(device->host_allocator, context);
      status = release_status;
    }
  }
  return status;
}
