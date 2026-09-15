// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/mcdm/legacy_context.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#include <stddef.h>
#include <string.h>
#include <windows.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/xdna/umd/mcdm/npu5_legacy_bootstrap_image.h"

// Fixed header of the installed NPU5 legacy context-private ABI.
typedef struct amdf_windows_xdna_legacy_context_header_t {
  // UUID copied from the provider-owned compatibility image.
  uint8_t xclbin_uuid[16];
  // Unresolved legacy state required before context policy.
  uint8_t reserved_0010[0x18];
  // Context quality-of-service priority; zero selects the default.
  uint32_t quality_of_service_priority;
  // Unresolved legacy state preceding the returned command cookie.
  uint8_t reserved_002c[0x14];
  // Command-aperture cookie written by the driver during context creation.
  uint32_t command_aperture_cookie;
  // Unresolved legacy state adjoining the command-aperture cookie.
  uint32_t reserved_0044;
  // Base of the driver's context command aperture.
  uint64_t command_aperture_base;
  // Unresolved NPU5 legacy value fixed at 0x48.
  uint64_t opaque_0050;
  // Number of bytes following offset 0x80 in the complete record.
  uint64_t bytes_after_0080;
  // Identifier of the process creating the context.
  uint64_t process_id;
  // Unresolved legacy state preceding offset 0x80.
  uint8_t reserved_0068[0x18];
  // Unresolved NPU5 legacy value fixed at one.
  uint64_t opaque_0080;
  // Unresolved legacy state preceding context command storage metadata.
  uint8_t reserved_0088[0x40];
  // Size in bytes of the context command storage expected by the driver.
  uint64_t context_command_storage_size;
  // Size in bytes of the embedded compatibility image.
  uint64_t xclbin_size;
  // Number of bytes following virtual record offset 0x138.
  uint64_t bytes_after_0138;
  // Combined embedded-image and trailing-record size.
  uint64_t embedded_payload_size;
} amdf_windows_xdna_legacy_context_header_t;

// Fixed trailing record following the variably sized compatibility image.
typedef struct amdf_windows_xdna_legacy_context_tail_t {
  // Legacy kernel name and driver-required terminal discriminator.
  char kernel_name[0x40];
  // Register address range in bytes of the compatibility kernel.
  uint64_t kernel_register_byte_length;
  // Number of arguments declared by the compatibility kernel.
  uint64_t kernel_argument_count;
  // Unresolved legacy state preceding the kernel identifier.
  uint8_t reserved_0050[8];
  // Legacy kernel identifier used only by the compatibility envelope.
  uint64_t kernel_id;
  // Unresolved program-independent legacy context state.
  uint8_t reserved_0060[0x300];
  // Native partition-admission inputs, not achieved placement or geometry.
  struct {
    // Operations per AIE cycle used for native quality-of-service accounting.
    uint32_t operations_per_cycle;
    // Number of candidate physical starting columns following this header.
    uint32_t start_column_count;
    // Requested partition width in columns.
    uint32_t column_count;
    // Candidate physical starting columns supplied to native admission.
    uint32_t start_columns[4];
  } partition;
} amdf_windows_xdna_legacy_context_tail_t;

#define AMDF_WINDOWS_XDNA_LEGACY_CONTEXT_TAIL_SIZE                \
  (offsetof(amdf_windows_xdna_legacy_context_tail_t, partition) + \
   sizeof(((amdf_windows_xdna_legacy_context_tail_t*)0)->partition))

_Static_assert(sizeof(amdf_windows_xdna_legacy_context_header_t) == 0xE8,
               "legacy context header layout must match the NPU5 ABI");
_Static_assert(offsetof(amdf_windows_xdna_legacy_context_header_t,
                        command_aperture_cookie) == 0x40,
               "legacy command cookie offset must match the NPU5 ABI");
_Static_assert(AMDF_WINDOWS_XDNA_LEGACY_CONTEXT_TAIL_SIZE == 0x37C,
               "legacy context tail layout must match the NPU5 ABI");

// Metadata-only native envelope. The driver reaches admission through
// record[0x40] + 0x78 + record[record[0x40] + 0x68]. The skipped-payload
// length is zero: the kernel consumes neither xclbin nor kernel records.
typedef struct amdf_windows_xdna_metadata_context_t {
  // UUID identifying the native bootstrap rather than application code.
  uint8_t bootstrap_uuid[16];
  // Zero selects default native quality-of-service policy.
  uint8_t quality_of_service[0x20];
  // Driver-written context ID, including zero.
  uint32_t command_aperture_cookie;
  // Native alignment preceding the firmware aperture address.
  uint32_t reserved_0034;
  // Firmware-visible base requested for this context's instruction window.
  uint64_t command_aperture_base;
  // Byte offset of the native metadata-container header within this record.
  uint64_t container_byte_offset;
  // Number of bytes following native container offset 0x70.
  uint64_t bytes_after_0070;
  // Process creating the native context.
  uint64_t process_id;
  // Unused native container state; no image or kernel-description payload.
  uint8_t reserved_0058[0x58];
  // Bytes skipped before the admission metadata, zero in this representation.
  uint64_t skipped_payload_byte_length;
  // Unused native container size field.
  uint64_t reserved_00b8;
  // Native partition name, empty for program-independent admission.
  uint8_t partition_name[0x40];
  // Nominal accounting from the selected native bootstrap.
  uint32_t operations_per_cycle;
  // A nonempty candidate list is required even though the driver replaces it.
  uint32_t start_column_count;
  // Requested logical partition extent.
  uint32_t column_count;
  // Admission input only; this is not a binding placement guarantee.
  uint32_t first_start_column;
} amdf_windows_xdna_metadata_context_t;

_Static_assert(sizeof(amdf_windows_xdna_metadata_context_t) == 0x110,
               "metadata context record must match its native ABI");
_Static_assert(offsetof(amdf_windows_xdna_metadata_context_t,
                        skipped_payload_byte_length) == 0xB0,
               "metadata skip length must match the native locator");
_Static_assert(offsetof(amdf_windows_xdna_metadata_context_t, partition_name) ==
                   0xC0,
               "admission metadata must follow the container header");

static amdf_status_t amdf_windows_xdna_metadata_context_build(
    const amdf_xdna_bootstrap_t* bootstrap, uint32_t partition_column_count,
    uint32_t first_start_column, amdf_allocator_t host_allocator,
    uint8_t** out_data, uint32_t* out_data_size) {
  amdf_windows_xdna_metadata_context_t* data = NULL;
  const amdf_status_t status = amdf_calloc(
      host_allocator, sizeof(*data),
      amdf_alignof(amdf_windows_xdna_metadata_context_t), (void**)&data);
  if (!amdf_status_is_ok(status)) return status;
  memcpy(data->bootstrap_uuid, bootstrap->context.uuid,
         sizeof(data->bootstrap_uuid));
  data->command_aperture_base = UINT64_C(0x04000000);
  data->container_byte_offset = 0x48;
  data->bytes_after_0070 = sizeof(*data) - 0x70;
  data->process_id = GetCurrentProcessId();
  data->operations_per_cycle = bootstrap->context.operations_per_cycle;
  data->start_column_count = 1;
  data->column_count = partition_column_count;
  data->first_start_column = first_start_column;
  *out_data = (uint8_t*)data;
  *out_data_size = sizeof(*data);
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_windows_xdna_legacy_context_build(
    const amdf_windows_xdna_native_abi_t* abi,
    const amdf_xdna_bootstrap_t* bootstrap, uint32_t partition_column_count,
    uint32_t first_start_column, amdf_allocator_t host_allocator,
    uint8_t** out_data, uint32_t* out_data_size) {
  if (out_data == NULL || out_data_size == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (abi->context_encoding == AMDF_WINDOWS_XDNA_CONTEXT_ENCODING_METADATA) {
    return amdf_windows_xdna_metadata_context_build(
        bootstrap, partition_column_count, first_start_column, host_allocator,
        out_data, out_data_size);
  }
  const size_t xclbin_uuid_offset = 0x1A0;
  if (amdf_windows_xdna_npu5_legacy_bootstrap_image_size <
          xclbin_uuid_offset + 16 ||
      memcmp(amdf_windows_xdna_npu5_legacy_bootstrap_image, "xclbin2\0", 8) !=
          0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  // The retained container is qualified only for its matching bootstrap. A
  // different target does not inherit it merely by selecting the same wire ABI.
  if (memcmp(bootstrap->context.uuid,
             amdf_windows_xdna_npu5_legacy_bootstrap_image + xclbin_uuid_offset,
             sizeof(bootstrap->context.uuid)) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }

  const size_t total_size = sizeof(amdf_windows_xdna_legacy_context_header_t) +
                            amdf_windows_xdna_npu5_legacy_bootstrap_image_size +
                            AMDF_WINDOWS_XDNA_LEGACY_CONTEXT_TAIL_SIZE;
  if (total_size > UINT32_MAX) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  uint8_t* data = NULL;
  const amdf_status_t allocation_status =
      amdf_calloc(host_allocator, total_size, amdf_max_align_t, (void**)&data);
  if (!amdf_status_is_ok(allocation_status)) return allocation_status;

  amdf_windows_xdna_legacy_context_header_t* header =
      (amdf_windows_xdna_legacy_context_header_t*)data;
  memcpy(header->xclbin_uuid,
         amdf_windows_xdna_npu5_legacy_bootstrap_image + xclbin_uuid_offset,
         sizeof(header->xclbin_uuid));
  header->command_aperture_base = UINT64_C(0x04000000);
  header->opaque_0050 = UINT64_C(0x48);
  header->bytes_after_0080 = total_size - 0x80;
  header->process_id = GetCurrentProcessId();
  header->opaque_0080 = 1;
  header->context_command_storage_size = UINT64_C(0x1000);
  header->xclbin_size = amdf_windows_xdna_npu5_legacy_bootstrap_image_size;
  header->bytes_after_0138 = total_size - 0x138;
  header->embedded_payload_size = total_size - sizeof(*header);

  uint8_t* image_data = data + sizeof(*header);
  memcpy(image_data, amdf_windows_xdna_npu5_legacy_bootstrap_image,
         amdf_windows_xdna_npu5_legacy_bootstrap_image_size);

  amdf_windows_xdna_legacy_context_tail_t tail = {0};
  static const char kernel_name[] = "MLIR_AIE";
  memcpy(tail.kernel_name, kernel_name, sizeof(kernel_name));
  tail.kernel_name[0x3F] = '0';
  tail.kernel_register_byte_length = UINT64_C(0x10000);
  tail.kernel_argument_count = 8;
  tail.kernel_id = UINT64_C(0x901);
  tail.partition.operations_per_cycle = bootstrap->context.operations_per_cycle;
  tail.partition.start_column_count = 4;
  tail.partition.column_count = partition_column_count;
  tail.partition.start_columns[0] = first_start_column;
  tail.partition.start_columns[1] = 2;
  tail.partition.start_columns[2] = 3;
  tail.partition.start_columns[3] = 4;
  memcpy(image_data + amdf_windows_xdna_npu5_legacy_bootstrap_image_size, &tail,
         AMDF_WINDOWS_XDNA_LEGACY_CONTEXT_TAIL_SIZE);

  *out_data = data;
  *out_data_size = (uint32_t)total_size;
  return AMDF_STATUS_OK;
}

uint32_t amdf_windows_xdna_legacy_context_query_command_aperture_cookie(
    const amdf_windows_xdna_native_abi_t* abi, const uint8_t* data) {
  uint32_t cookie;
  memcpy(&cookie, data + abi->context_cookie_byte_offset, sizeof(cookie));
  return cookie;
}
