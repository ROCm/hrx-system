// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/context.h"

#include <string.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/xdna/umd/mcdm/context.h"
#include "libamdf/src/xdna/umd/mcdm/kernel_execution.h"

// Fixed direct-width native context record. Its kernel buffer is native
// context storage, not an application program or a per-submission BO list.
typedef struct amdf_windows_xdna_direct_context_t {
  // Optional image identity; zero for a program-independent partition.
  uint8_t uuid[16];
  // Zero selects the native default quality-of-service policy.
  uint8_t quality_of_service[0x30];
  // Reserved context configuration, zero for the ordinary native path.
  uint32_t reserved_0040;
  // Context ID returned by native creation, including zero.
  uint32_t command_aperture_cookie;
  // Size of the native instruction aperture in bytes.
  uint64_t command_aperture_byte_length;
  // Process creating the native context.
  uint32_t process_id;
  // Requested logical partition width in columns.
  uint32_t column_count;
  // Reserved native placement and proxy configuration; all zero.
  uint32_t reserved_0058[4];
  // Native allocation handle retained by the context owner.
  uint64_t kernel_buffer_allocation;
  // Byte offset within the kernel buffer allocation.
  uint32_t kernel_buffer_byte_offset;
  // Byte length of the kernel buffer.
  uint32_t kernel_buffer_byte_length;
  // Locked CPU pointer retained through native context destruction.
  uint64_t kernel_buffer_host_address;
  // Optional native configuration, zero on the direct-width path.
  uint8_t reserved_0080[0x20];
} amdf_windows_xdna_direct_context_t;

_Static_assert(sizeof(amdf_windows_xdna_direct_context_t) == 0xA0,
               "direct context must match the native wire record");
_Static_assert(offsetof(amdf_windows_xdna_direct_context_t,
                        kernel_buffer_allocation) == 0x68,
               "direct context buffer must match the native wire offset");

// Baseline metadata admission. The native decoder locates the partition as
// record + container_byte_offset + 0x78 + skipped_payload_byte_length.
typedef struct amdf_windows_xdna_compact_metadata_context_t {
  // Identity of the target bootstrap, independent of application code.
  uint8_t uuid[16];
  // Zero selects the native default quality-of-service policy.
  uint8_t quality_of_service[0x20];
  // Context ID returned by native creation, including zero.
  uint32_t command_aperture_cookie;
  // Native alignment preceding the firmware aperture address.
  uint32_t reserved_0034;
  // Size of the native instruction aperture in bytes.
  uint64_t command_aperture_byte_length;
  // Byte offset locating the native metadata container within this record.
  uint64_t container_byte_offset;
  // Number of bytes following native record offset 0x70.
  uint64_t bytes_after_0070;
  // Process creating the native context.
  uint32_t process_id;
  // Zero selects ordinary native context creation.
  uint32_t reserved_0054;
  // Unused container state; no image or kernel-description payload is supplied.
  uint8_t reserved_0058[0x58];
  // Bytes skipped before partition metadata, zero in this representation.
  uint64_t skipped_payload_byte_length;
  // Unused native container size field.
  uint64_t reserved_00b8;
  // Native partition name, empty for program-independent admission.
  uint8_t partition_name[0x40];
  // Nominal accounting from the target bootstrap.
  uint32_t operations_per_cycle;
  // A nonempty candidate list is required; the driver chooses placement.
  uint32_t start_column_count;
  // Requested logical partition width in columns.
  uint32_t column_count;
  // Admission input only, not a binding placement guarantee.
  uint32_t first_start_column;
} amdf_windows_xdna_compact_metadata_context_t;

_Static_assert(sizeof(amdf_windows_xdna_compact_metadata_context_t) == 0x110,
               "compact metadata context must match the native wire record");
_Static_assert(offsetof(amdf_windows_xdna_compact_metadata_context_t,
                        column_count) == 0x108,
               "partition width must match the compact metadata locator");

// Metadata partition admission. The native decoder locates the partition as
// record + container_byte_offset + 0xA0 + skipped_payload_byte_length. There
// is no embedded image or kernel metadata between the container and partition.
typedef struct amdf_windows_xdna_metadata_context_t {
  // Identity of the target bootstrap, independent of application code.
  uint8_t uuid[16];
  // Zero selects the native default quality-of-service policy.
  uint8_t quality_of_service[0x30];
  // Context ID returned by native creation, including zero.
  uint32_t command_aperture_cookie;
  // Native alignment preceding the firmware aperture address.
  uint32_t reserved_0044;
  // Size of the native instruction aperture in bytes.
  uint64_t command_aperture_byte_length;
  // Byte offset locating the native metadata container within this record.
  uint64_t container_byte_offset;
  // Number of bytes following native record offset 0x80.
  uint64_t bytes_after_0080;
  // Process creating the native context.
  uint32_t process_id;
  // Zero selects ordinary native context creation.
  uint32_t reserved_0064;
  // Unused container state; no image or kernel-description payload is supplied.
  uint8_t reserved_0068[0x70];
  // Bytes skipped before partition metadata, zero in this representation.
  uint64_t skipped_payload_byte_length;
  // Unused native container size field.
  uint64_t reserved_00e0;
  // Native partition name, empty for program-independent admission.
  uint8_t partition_name[0x40];
  // Nominal accounting from the target bootstrap.
  uint32_t operations_per_cycle;
  // A nonempty candidate list is required; the driver chooses placement.
  uint32_t start_column_count;
  // Requested logical partition width in columns.
  uint32_t column_count;
  // Admission input only, not a binding placement guarantee.
  uint32_t first_start_column;
} amdf_windows_xdna_metadata_context_t;

_Static_assert(sizeof(amdf_windows_xdna_metadata_context_t) == 0x138,
               "metadata context must match the native wire record");
_Static_assert(offsetof(amdf_windows_xdna_metadata_context_t, column_count) ==
                   0x130,
               "partition width must match the native metadata locator");

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
  const amdf_status_t buffer_status =
      amdf_windows_xdna_private_allocation_destroy(&context->kernel_buffer);
  if (!amdf_status_is_ok(buffer_status)) return buffer_status;
  const amdf_allocator_t host_allocator = context->device->host_allocator;
  amdf_free(host_allocator, context);
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_xdna_umd_context_create(
    amdf_xdna_umd_device_t* device,
    const amdf_xdna_context_create_info_t* create_info,
    amdf_xdna_umd_context_t** out_context,
    amdf_xdna_umd_context_result_t* out_result) {
  const amdf_xdna_device_profile_t* profile = device->profile;
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

  amdf_windows_xdna_adapter_info_t adapter_info = {0};
  amdf_status_t status = amdf_windows_xdna_adapter_info_query(
      device->kmt, device->adapter, &adapter_info);
  if (!amdf_status_is_ok(status)) return status;
  if (!amdf_kmt_api_supports_memory(device->kmt) || device->kmt->lock == NULL ||
      device->kmt->unlock == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_xdna_umd_context_t* context = NULL;
  status = amdf_calloc(device->host_allocator, sizeof(*context),
                       amdf_alignof(amdf_xdna_umd_context_t), (void**)&context);
  if (!amdf_status_is_ok(status)) return status;
  context->device = device;
  context->adapter_info = adapter_info;

  const bool direct =
      adapter_info.protocol == AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT;
  if (direct) {
    const amdf_windows_xdna_private_allocation_descriptor_t descriptor = {
        .requested_byte_length = 4096,
        .allocation_byte_length = 4096,
        .type = 0x332C,
        .policy = 2,
        .xcl_flags = 0x02000000,
        .flags =
            AMDF_WINDOWS_XDNA_PRIVATE_ALLOCATION_FLAG_DEVICE_ADDRESS |
            (adapter_info.shared_kernel_buffers
                 ? AMDF_WINDOWS_XDNA_PRIVATE_ALLOCATION_FLAG_SHARED_RESOURCE
                 : 0),
    };
    amdf_windows_xdna_private_allocation_initialize(device, &descriptor,
                                                    &context->kernel_buffer);
    status =
        amdf_windows_xdna_private_allocation_realize(&context->kernel_buffer);
    if (amdf_status_is_ok(status)) {
      status =
          amdf_windows_xdna_private_allocation_lock(&context->kernel_buffer);
    }
  }
  union {
    // Direct admission with a retained native kernel buffer.
    amdf_windows_xdna_direct_context_t direct;
    // Baseline partition-metadata admission without private adapter
    // information.
    amdf_windows_xdna_compact_metadata_context_t compact_metadata;
    // Partition-metadata admission without an image container.
    amdf_windows_xdna_metadata_context_t metadata;
  } context_data = {0};
  uint32_t context_data_size;
  if (direct) {
    context_data.direct = (amdf_windows_xdna_direct_context_t){
        .command_aperture_byte_length = AMDF_WINDOWS_XDNA_PRIVATE_APERTURE_SIZE,
        .process_id = GetCurrentProcessId(),
        .column_count = create_info->logical_column_count,
        .kernel_buffer_allocation = context->kernel_buffer.allocation,
        .kernel_buffer_byte_length =
            (uint32_t)context->kernel_buffer.descriptor.allocation_byte_length,
        .kernel_buffer_host_address =
            (uintptr_t)context->kernel_buffer.host_pointer,
    };
    context_data_size = sizeof(context_data.direct);
  } else if (adapter_info.protocol ==
             AMDF_WINDOWS_XDNA_PROTOCOL_METADATA_COMPACT) {
    context_data.compact_metadata =
        (amdf_windows_xdna_compact_metadata_context_t){
            .command_aperture_byte_length =
                AMDF_WINDOWS_XDNA_PRIVATE_APERTURE_SIZE,
            .container_byte_offset = 0x48,
            .bytes_after_0070 = sizeof(context_data.compact_metadata) - 0x70,
            .process_id = GetCurrentProcessId(),
            .operations_per_cycle =
                profile->bootstrap->context.operations_per_cycle,
            .start_column_count = 1,
            .column_count = create_info->logical_column_count,
            .first_start_column = profile->info->array.column_origin,
        };
    memcpy(context_data.compact_metadata.uuid, profile->bootstrap->context.uuid,
           sizeof(context_data.compact_metadata.uuid));
    context_data_size = sizeof(context_data.compact_metadata);
  } else {
    context_data.metadata = (amdf_windows_xdna_metadata_context_t){
        .command_aperture_byte_length = AMDF_WINDOWS_XDNA_PRIVATE_APERTURE_SIZE,
        .container_byte_offset = 0x48,
        .bytes_after_0080 = sizeof(context_data.metadata) - 0x80,
        .process_id = GetCurrentProcessId(),
        .operations_per_cycle =
            profile->bootstrap->context.operations_per_cycle,
        .start_column_count = 1,
        .column_count = create_info->logical_column_count,
        .first_start_column = profile->info->array.column_origin,
    };
    memcpy(context_data.metadata.uuid, profile->bootstrap->context.uuid,
           sizeof(context_data.metadata.uuid));
    context_data_size = sizeof(context_data.metadata);
  }

  D3DKMT_CREATECONTEXTVIRTUAL create = {0};
  create.hDevice = device->device;
  create.NodeOrdinal = 0;
  create.EngineAffinity = 1;
  create.Flags.HwQueueSupported = 1;
  create.pPrivateDriverData = &context_data;
  create.PrivateDriverDataSize = context_data_size;
  create.ClientHint = (D3DKMT_CLIENTHINT)25;
  if (amdf_status_is_ok(status)) {
    status = amdf_kmt_make_status(device->kmt->create_context_virtual(&create));
  }
  if (amdf_status_is_ok(status)) {
    context->handle = create.hContext;
    if (context->handle == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    } else {
      switch (adapter_info.protocol) {
        case AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT:
          context->command_aperture_cookie =
              context_data.direct.command_aperture_cookie;
          break;
        case AMDF_WINDOWS_XDNA_PROTOCOL_METADATA:
          context->command_aperture_cookie =
              context_data.metadata.command_aperture_cookie;
          break;
        case AMDF_WINDOWS_XDNA_PROTOCOL_METADATA_COMPACT:
          context->command_aperture_cookie =
              context_data.compact_metadata.command_aperture_cookie;
          break;
      }
      if (context->command_aperture_cookie > UINT8_MAX) {
        status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
      }
    }
  }

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
      // No execution work was accepted. A failed native release leaks its
      // backing; retaining unreachable host bookkeeping cannot recover it.
      amdf_free(device->host_allocator, context);
      status = release_status;
    }
  }
  return status;
}
